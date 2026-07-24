#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, inverse_tanh, get_expon_lr_func, build_rotation, build_rotation_4d, build_scaling_rotation_4d, inverse_sigmoid_opa
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2, distCUDA2b
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import sh_channels_4d, eval_sh
from utils.transformation_util import quaternion_to_matrix
import gc
from scene.NVDIFFREC import create_trainable_env_rnd
import nvdiffrast.torch



def _parse_frame_filter(frame_filter):
    if frame_filter is None or frame_filter == "":
        return None
    if isinstance(frame_filter, (list, tuple, set)):
        return {int(value) for value in frame_filter}
    frames = set()
    for part in str(frame_filter).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            frames.update(range(int(start), int(end) + 1))
        else:
            frames.add(int(part))
    return frames


def _parse_sph_band_spec(spec):
    """Parse a hierarchy band spec 'channels@keyframes[,channels@keyframes...]',
    e.g. '4@5,3@15,1@30' -> [(4, 5), (3, 15), (1, 30)]."""
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        return [(int(ch), int(key)) for ch, key in spec]
    bands = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        channels, keyframes = part.split("@", 1)
        bands.append((int(channels), int(keyframes)))
    return bands

class SphMipEncoding(nn.Module):
    def __init__(
        self,
        n_levels: int = 8,
        plane_size: int = 512,
        #feature_dim: int = 16,
        feature_dim: int = 16,
        Sn: int = 1,
        dim: int = 1,
        rand_init: bool = False,
        time_min: float = 0.0,
        time_max: float = 1.0,
        residual_keyframes: int = 0,
        residual_start_iteration: int = 0,
        residual_bands=None,
        parity_bands=None,
        sliding_bands=None,
        sliding_window: int = 16,
    ):
        super(SphMipEncoding, self).__init__()
        self.n_levels = n_levels
        self.plane_size = plane_size
        self.time_min = float(time_min)
        self.time_max = float(time_max)
        self.residual_start_iteration = int(residual_start_iteration)
        
        self.register_parameter("fm", nn.Parameter(torch.zeros(Sn, dim, plane_size, 2*plane_size, feature_dim)),)

        self.fm_residual = None
        if residual_keyframes > 0:
            self._create_residual_keyframes(residual_keyframes)

        if rand_init:
            self.init_parameters()

        self.fm_residual_bands = None
        self.residual_band_spec = []
        parsed_bands = _parse_sph_band_spec(residual_bands)
        if parsed_bands:
            self._create_residual_bands(parsed_bands)

        self.fm_parity = None
        self.parity_spec = None
        parsed_parity = _parse_sph_band_spec(parity_bands)
        if parsed_parity:
            self._create_parity_keyframes(*parsed_parity[0])

        self.fm_sliding = None
        self.sliding_spec = None
        self.sliding_window = max(1, int(sliding_window))
        parsed_sliding = _parse_sph_band_spec(sliding_bands)
        if parsed_sliding:
            self._create_sliding_keyframes(*parsed_sliding[0], self.sliding_window)

    def _create_residual_keyframes(self, residual_keyframes):
        base = self.fm
        self.fm_residual = nn.ParameterList([
            nn.Parameter(torch.zeros(base.shape[1:], dtype=base.dtype, device=base.device))
            for _ in range(int(residual_keyframes))
        ])
        per_map_mb = base.shape[1:].numel() * base.element_size() / (1024 ** 2)
        print(f"[SphMipEncoding] Created {int(residual_keyframes)} time-switched residual keyframes "
              f"({per_map_mb:.1f} MB each, zero-init).")

    def _create_residual_bands(self, bands):
        """Hierarchical time bands: the trailing channel groups of the base map
        are represented directly by keyframe stacks at different temporal
        frequencies (linearly interpolated in time), while the leading channels
        stay static (base map only).  Each band keyframe is initialized from the
        base map's corresponding channels so the encoded function is unchanged
        at creation time (zeros for a freshly built encoding)."""
        base = self.fm
        feature_dim = base.shape[-1]
        bands = [(int(ch), max(1, int(key))) for ch, key in bands if int(ch) > 0]
        total_band_ch = sum(ch for ch, _ in bands)
        if total_band_ch > feature_dim:
            raise ValueError(f"[SphMipEncoding] Hierarchy bands use {total_band_ch} channels "
                             f"but feature_dim is only {feature_dim}.")
        self.residual_band_spec = bands
        band_modules = []
        channel = feature_dim - total_band_ch
        for ch, key_count in bands:
            base_slice = base.detach()[0, ..., channel:channel + ch].clone()
            band_modules.append(nn.ParameterList([
                nn.Parameter(base_slice.clone())
                for _ in range(key_count)
            ]))
            channel += ch
        self.fm_residual_bands = nn.ModuleList(band_modules)
        self.bands_are_direct = True
        per_channel_map_mb = base.shape[1:-1].numel() * base.element_size() / (1024 ** 2)
        total_mb = sum(ch * key for ch, key in bands) * per_channel_map_mb
        layout = ", ".join(f"{ch}ch@{key}kf" for ch, key in bands)
        print(f"[SphMipEncoding] Hierarchical time bands: {feature_dim - total_band_ch} static ch + "
              f"{layout} (direct keyframes, lerp, {total_mb:.1f} MB total).")

    def _upgrade_residual_bands_to_direct(self):
        """Older checkpoints stored the time bands as residuals added onto the
        base map's trailing channels; fold the base map into every keyframe once
        so the stacks become direct band values.  Idempotent."""
        if getattr(self, "fm_residual_bands", None) is None or not getattr(self, "residual_band_spec", []):
            return
        if getattr(self, "bands_are_direct", False):
            return
        with torch.no_grad():
            total_band_ch = sum(ch for ch, _ in self.residual_band_spec)
            channel = self.fm.shape[-1] - total_band_ch
            for band_params, (ch, _) in zip(self.fm_residual_bands, self.residual_band_spec):
                base_slice = self.fm[0, ..., channel:channel + ch]
                for keyframe in band_params:
                    keyframe.add_(base_slice)
                channel += ch
        self.bands_are_direct = True
        print("[SphMipEncoding] Upgraded residual time bands to direct band keyframes (base map folded in).")

    def ensure_residual_hierarchy(self, residual_bands, time_min=None, time_max=None, start_iteration=None):
        """Idempotent setup of hierarchical time-band keyframes.

        Also upgrades dir_encoding modules unpickled from older checkpoints that
        predate the feature or that stored the bands as residuals on top of the
        base map, and refreshes the time range mapping."""
        if time_min is not None:
            self.time_min = float(time_min)
        if time_max is not None:
            self.time_max = float(time_max)
        if start_iteration is not None:
            self.residual_start_iteration = int(start_iteration)
        elif not hasattr(self, "residual_start_iteration"):
            self.residual_start_iteration = 0
        bands = [(int(ch), max(1, int(key))) for ch, key in _parse_sph_band_spec(residual_bands) if int(ch) > 0]
        if not bands:
            return
        existing_spec = getattr(self, "residual_band_spec", None)
        if existing_spec == bands and getattr(self, "fm_residual_bands", None) is not None:
            self._upgrade_residual_bands_to_direct()
            return
        if existing_spec:
            print(f"[SphMipEncoding] Hierarchy band spec changed {existing_spec} -> {bands}; parameters reset.")
        flat_residual = getattr(self, "fm_residual", None)
        if flat_residual is not None and len(flat_residual) > 0:
            print("[SphMipEncoding] Warning: flat residual keyframes present alongside hierarchical bands; both will be applied.")
        self._create_residual_bands(bands)

    def _get_band_keyframe_weights(self, timestamp, key_count):
        if key_count <= 1:
            return 0, 0, 0.0
        if torch.is_tensor(timestamp):
            time_value = float(timestamp.detach().reshape(-1)[0].item())
        else:
            time_value = float(timestamp)
        time_min = float(getattr(self, "time_min", 0.0))
        time_max = float(getattr(self, "time_max", 1.0))
        denom = max(time_max - time_min, 1.0e-6)
        normalized_t = min(max((time_value - time_min) / denom, 0.0), 1.0)
        scaled_t = normalized_t * (key_count - 1)
        key0 = min(int(math.floor(scaled_t)), key_count - 2)
        return key0, key0 + 1, scaled_t - key0

    def _compose_hierarchical_bands(self, fm, timestamp):
        """Assemble the time-dependent feature map: the leading channels come
        from the static base map, while each trailing channel band is read
        directly from its own keyframe stack, linearly interpolated between the
        two nearest keyframes."""
        spec = self.residual_band_spec
        total_band_ch = sum(ch for ch, _ in spec)
        pieces = [fm[..., :fm.shape[-1] - total_band_ch]]
        for band_params, (ch, key_count) in zip(self.fm_residual_bands, spec):
            key0, key1, alpha = self._get_band_keyframe_weights(timestamp, key_count)
            if key1 == key0 or alpha <= 0.0:
                band_fm = band_params[key0]
            elif alpha >= 1.0:
                band_fm = band_params[key1]
            else:
                band_fm = band_params[key0] * (1.0 - alpha) + band_params[key1] * alpha
            pieces.append(band_fm)
        return torch.cat(pieces, dim=-1)

    def _create_parity_keyframes(self, channels, keyframes):
        """Parity-slot dynamic keyframes: a sparse stack of C-channel env maps at K
        keyframes.  Instead of lerping the maps, the two keyframes adjacent to the
        query time are hat-weighted and routed to two fixed decoder slots (even
        keyframe index -> slot A, odd -> slot B) so the light MLP sees both
        endpoints plus the blend weights while its input stays continuous in
        time (the outgoing map's weight reaches zero exactly when slot contents
        swap at a crossing)."""
        base = self.fm
        channels = int(channels)
        keyframes = max(1, int(keyframes))
        self.parity_spec = (channels, keyframes)
        self.fm_parity = nn.ParameterList([
            nn.Parameter(torch.zeros(*base.shape[1:-1], channels, dtype=base.dtype, device=base.device))
            for _ in range(keyframes)
        ])
        per_map_mb = base.shape[1:-1].numel() * channels * base.element_size() / (1024 ** 2)
        print(f"[SphMipEncoding] Parity-slot dynamic keyframes: {channels}ch @ {keyframes}kf "
              f"({per_map_mb:.1f} MB each, {per_map_mb * keyframes:.1f} MB total, zero-init).")

    def ensure_parity_keyframes(self, parity_bands, time_min=None, time_max=None):
        """Idempotent setup of parity-slot dynamic keyframes.  Also upgrades
        dir_encoding modules unpickled from checkpoints that predate the feature
        and refreshes the time range mapping."""
        if time_min is not None:
            self.time_min = float(time_min)
        if time_max is not None:
            self.time_max = float(time_max)
        parsed = _parse_sph_band_spec(parity_bands)
        if not parsed:
            return
        channels, keyframes = int(parsed[0][0]), max(1, int(parsed[0][1]))
        existing = getattr(self, "fm_parity", None)
        if existing is not None and len(existing) > 0 and getattr(self, "parity_spec", None) == (channels, keyframes):
            return
        if existing is not None and len(existing) > 0:
            print(f"[SphMipEncoding] Parity keyframe spec changed "
                  f"{getattr(self, 'parity_spec', None)} -> {(channels, keyframes)}; parameters reset.")
        self._create_parity_keyframes(channels, keyframes)

    @property
    def parity_channels(self):
        fm_parity = getattr(self, "fm_parity", None)
        if fm_parity is None or len(fm_parity) == 0:
            return 0
        return fm_parity[0].shape[-1]

    def get_parity_slot_weights(self, timestamp):
        """Hat weights routed to parity slots: even-index keyframes always occupy
        slot A, odd-index keyframes slot B.  Returns (slot_a_idx, slot_b_idx,
        w_a, w_b).  Both the weighted slot features and the weight scalars are
        continuous across keyframe crossings."""
        fm_parity = getattr(self, "fm_parity", None)
        key_count = 0 if fm_parity is None else len(fm_parity)
        if key_count == 0 or timestamp is None:
            return 0, 0, 0.0, 0.0
        if key_count == 1:
            return 0, 0, 1.0, 0.0
        key0, key1, alpha = self._get_band_keyframe_weights(timestamp, key_count)
        if key0 % 2 == 0:
            return key0, key1, 1.0 - alpha, alpha
        return key1, key0, alpha, 1.0 - alpha

    def _create_sliding_keyframes(self, channels, keyframes, window):
        """Sliding-window dynamic keyframes: a stack of C-channel env maps at K
        keyframes decoded through W fixed slots.  Slot s always holds the
        keyframe with index congruent to s (mod W) nearest to the query time,
        tent-weighted with half-width W/2 keyframe intervals, so up to W
        keyframes are visible to the light MLP simultaneously and a slot swaps
        its occupant exactly when its weight is zero (contents change one
        keyframe at a time; the weighted features stay continuous in time)."""
        base = self.fm
        channels = int(channels)
        keyframes = max(1, int(keyframes))
        window = max(1, int(window))
        self.sliding_spec = (channels, keyframes)
        self.sliding_window = window
        self.fm_sliding = nn.ParameterList([
            nn.Parameter(torch.zeros(*base.shape[1:-1], channels, dtype=base.dtype, device=base.device))
            for _ in range(keyframes)
        ])
        per_map_mb = base.shape[1:-1].numel() * channels * base.element_size() / (1024 ** 2)
        print(f"[SphMipEncoding] Sliding-window dynamic keyframes: {channels}ch @ {keyframes}kf, "
              f"{window} slots ({per_map_mb:.1f} MB each, {per_map_mb * keyframes:.1f} MB total, zero-init).")

    def ensure_sliding_keyframes(self, sliding_bands, sliding_window=None, time_min=None, time_max=None):
        """Idempotent setup of sliding-window dynamic keyframes.  Also upgrades
        dir_encoding modules unpickled from checkpoints that predate the feature
        and refreshes the time range mapping."""
        if time_min is not None:
            self.time_min = float(time_min)
        if time_max is not None:
            self.time_max = float(time_max)
        parsed = _parse_sph_band_spec(sliding_bands)
        if not parsed:
            return
        channels, keyframes = int(parsed[0][0]), max(1, int(parsed[0][1]))
        window = max(1, int(sliding_window if sliding_window is not None else getattr(self, "sliding_window", 16)))
        existing = getattr(self, "fm_sliding", None)
        if existing is not None and len(existing) > 0 and getattr(self, "sliding_spec", None) == (channels, keyframes):
            if int(getattr(self, "sliding_window", 0)) != window:
                print(f"[SphMipEncoding] Sliding window changed "
                      f"{getattr(self, 'sliding_window', None)} -> {window} (keyframe stack kept).")
                self.sliding_window = window
            return
        if existing is not None and len(existing) > 0:
            print(f"[SphMipEncoding] Sliding keyframe spec changed "
                  f"{getattr(self, 'sliding_spec', None)} -> {(channels, keyframes)}; parameters reset.")
        self._create_sliding_keyframes(channels, keyframes, window)

    @property
    def sliding_channels(self):
        fm_sliding = getattr(self, "fm_sliding", None)
        if fm_sliding is None or len(fm_sliding) == 0:
            return 0
        return fm_sliding[0].shape[-1]

    @property
    def sliding_slots(self):
        fm_sliding = getattr(self, "fm_sliding", None)
        if fm_sliding is None or len(fm_sliding) == 0:
            return 0
        return max(1, int(getattr(self, "sliding_window", 16)))

    def get_sliding_slot_weights(self, timestamp):
        """Sliding-window slot routing.  For each of the W slots, pick the
        keyframe with index congruent to the slot (mod W) nearest to the query
        time and tent-weight it with half-width W/2 keyframe intervals:
        alpha_s = max(0, 1 - |t_seg - k_s| / (W/2)).  Two keyframes sharing a
        slot are W intervals apart, so both can never be inside the open window
        at once, and an occupant swap happens exactly at zero weight -> the
        weighted slot features and the weights are continuous in time and only
        one slot changes occupant at a time.  Returns (indices, weights)."""
        fm_sliding = getattr(self, "fm_sliding", None)
        key_count = 0 if fm_sliding is None else len(fm_sliding)
        window = max(1, int(getattr(self, "sliding_window", 16)))
        if key_count == 0:
            return [], []
        if timestamp is None:
            return [0] * window, [0.0] * window
        if torch.is_tensor(timestamp):
            time_value = float(timestamp.detach().reshape(-1)[0].item())
        else:
            time_value = float(timestamp)
        time_min = float(getattr(self, "time_min", 0.0))
        time_max = float(getattr(self, "time_max", 1.0))
        denom = max(time_max - time_min, 1.0e-6)
        normalized_t = min(max((time_value - time_min) / denom, 0.0), 1.0)
        t_seg = normalized_t * (key_count - 1) if key_count > 1 else 0.0
        half_width = window / 2.0
        indices, weights = [], []
        for slot in range(window):
            if slot >= key_count:
                indices.append(0)
                weights.append(0.0)
                continue
            steps = round((t_seg - slot) / window)
            max_steps = (key_count - 1 - slot) // window
            steps = min(max(steps, 0), max_steps)
            key = slot + steps * window
            indices.append(key)
            weights.append(max(0.0, 1.0 - abs(t_seg - key) / half_width))
        return indices, weights

    def ensure_residual_keyframes(self, residual_keyframes, time_min=None, time_max=None, start_iteration=None):
        """Idempotent setup of time-switched residual keyframes.

        Also upgrades dir_encoding modules unpickled from older checkpoints that
        predate the residual feature, and refreshes the time range mapping."""
        if time_min is not None:
            self.time_min = float(time_min)
        if time_max is not None:
            self.time_max = float(time_max)
        if start_iteration is not None:
            self.residual_start_iteration = int(start_iteration)
        elif not hasattr(self, "residual_start_iteration"):
            self.residual_start_iteration = 0
        if residual_keyframes is None or residual_keyframes <= 0:
            return
        existing = getattr(self, "fm_residual", None)
        if existing is not None and len(existing) == int(residual_keyframes):
            return
        if existing is not None:
            print(f"[SphMipEncoding] Residual keyframe count changed {len(existing)} -> {int(residual_keyframes)}; parameters reset.")
        self._create_residual_keyframes(residual_keyframes)

    def get_time_segment_index(self, timestamp):
        fm_residual = getattr(self, "fm_residual", None)
        key_count = 0 if fm_residual is None else len(fm_residual)
        if key_count <= 1:
            return 0
        if torch.is_tensor(timestamp):
            time_value = float(timestamp.detach().reshape(-1)[0].item())
        else:
            time_value = float(timestamp)
        time_min = float(getattr(self, "time_min", 0.0))
        time_max = float(getattr(self, "time_max", 1.0))
        denom = max(time_max - time_min, 1.0e-6)
        normalized_t = min(max((time_value - time_min) / denom, 0.0), 1.0)
        return min(int(normalized_t * key_count), key_count - 1)

    def init_parameters(self) -> None:
        nn.init.uniform_(self.fm, -1e-2, 1e-2)

    def _sample_feature_map(self, fm, decomposed_x, level):
        padding_fm = torch.cat([fm[:, :, self.plane_size:, :], fm, fm[:, :, :self.plane_size, :]], dim=2)

        enc = nvdiffrast.torch.texture(
            padding_fm,
            decomposed_x,
            mip_level_bias=level * self.n_levels,
            boundary_mode="clamp",
            max_mip_level=self.n_levels - 1,
        )
        return enc.permute(1, 2, 0, 3).contiguous().view(decomposed_x.shape[0], -1)

    def _interpolate_feature_map(self, timestamp, device, dtype):
        key_count = self.fm.shape[0]
        if key_count <= 1 or timestamp is None:
            return None

        if torch.is_tensor(timestamp):
            time_value = timestamp.detach().to(device=device, dtype=dtype).reshape(-1).mean()
        else:
            time_value = torch.tensor(float(timestamp), device=device, dtype=dtype)

        time_min = float(getattr(self, "time_min", 0.0))
        time_max = float(getattr(self, "time_max", 1.0))
        denom = max(time_max - time_min, 1.0e-6)

        normalized_t = torch.clamp((time_value - time_min) / denom, 0.0, 1.0)
        scaled_t = normalized_t * (key_count - 1)
        key0 = int(torch.floor(scaled_t).item())
        key1 = min(key0 + 1, key_count - 1)
        alpha = (scaled_t - key0).to(dtype=self.fm.dtype)

        return torch.lerp(self.fm[key0], self.fm[key1], alpha)
        
    def forward(self, x, level, index=0, weight=False, timestamp=None, iteration=None):
        """
        x: [0,1], Nx3
        level: [0, max_level], Nx1
        """
        x = x.clone()
        x[..., 0] = x[..., 0] * 0.5 + 0.25
        
        decomposed_x = x.contiguous()
        
        level = torch.broadcast_to(level, decomposed_x.shape[:3]).contiguous()

        fm = self._interpolate_feature_map(timestamp, decomposed_x.device, decomposed_x.dtype)
        if fm is None:
            if torch.is_tensor(index):
                index = int(index.reshape(-1)[0].item())
            key_idx = max(0, min(int(index), self.fm.shape[0] - 1))
            fm = self.fm[key_idx]

        # Hard-switched (piecewise-constant in time) residual keyframes on top of the
        # static base map: no cross-keyframe interpolation, so moving reflections are
        # not blended between neighboring keyframes.
        fm_residual = getattr(self, "fm_residual", None)
        if (
            fm_residual is not None
            and len(fm_residual) > 0
            and timestamp is not None
            and (iteration is None or iteration >= getattr(self, "residual_start_iteration", 0))
        ):
            fm = fm + fm_residual[self.get_time_segment_index(timestamp)]

        # Hierarchical time bands: the trailing channel groups of the base map are
        # stored directly as keyframe stacks at different temporal frequencies and
        # linearly interpolated in time (piecewise-linear hat basis); the leading
        # channels stay static.  The bands ARE the feature map for those channels,
        # so they are active from the first iteration.
        fm_bands = getattr(self, "fm_residual_bands", None)
        if fm_bands is not None and len(fm_bands) > 0 and timestamp is not None:
            if not getattr(self, "bands_are_direct", False):
                self._upgrade_residual_bands_to_direct()
            fm = self._compose_hierarchical_bands(fm, timestamp)

        # Parity-slot dynamic keyframes: the two hat-weighted adjacent keyframe
        # maps are appended as extra channels (slot A = even keyframes, slot B =
        # odd) so the light MLP can decode both endpoints + blend weights instead
        # of consuming a pre-lerped map.  Weighting before sampling equals
        # weighting after (texture sampling is linear per channel).
        fm_parity = getattr(self, "fm_parity", None)
        if fm_parity is not None and len(fm_parity) > 0:
            slot_a_idx, slot_b_idx, w_a, w_b = self.get_parity_slot_weights(timestamp)
            slot_a = fm_parity[slot_a_idx] * w_a
            slot_b = fm_parity[slot_b_idx] * w_b
            fm = torch.cat([fm, slot_a, slot_b], dim=-1)

        # Sliding-window dynamic keyframes: the W tent-weighted slot maps (slot s
        # = nearest keyframe with index = s mod W) are appended as W*C extra
        # channels sampled in the same texture call; occupants change one at a
        # time and swap exactly at zero weight, so the sampled features stay
        # continuous in time.
        fm_sliding = getattr(self, "fm_sliding", None)
        if fm_sliding is not None and len(fm_sliding) > 0:
            slot_indices, slot_weights = self.get_sliding_slot_weights(timestamp)
            slot_maps = [fm_sliding[k] * w for k, w in zip(slot_indices, slot_weights)]
            fm = torch.cat([fm] + slot_maps, dim=-1)

        return self._sample_feature_map(fm, decomposed_x, level)


class SpecLightMLP(nn.Module):
    def __init__(self, base_dim: int, outer_dim: int, hidden_dim: int = 128, out_dim: int = 3):
        super(SpecLightMLP, self).__init__()
        self.base_dim = int(base_dim)
        self.outer_dim = int(outer_dim)
        self.fc1 = nn.Linear(self.base_dim, hidden_dim)
        self.fc_outer = nn.Linear(self.outer_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, base_feature, sph_outer_feature=None):
        # Compatibility path: accepts concatenated input from older call sites.
        if sph_outer_feature is None and base_feature.shape[-1] == self.base_dim + self.outer_dim:
            base_feature, sph_outer_feature = torch.split(base_feature, [self.base_dim, self.outer_dim], dim=-1)

        x = self.act(self.fc1(base_feature))
        if sph_outer_feature is not None:
            x = x + self.fc_outer(sph_outer_feature)
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.fc_out(x)
    

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    if 'red' in vertices and 'green' in vertices and 'blue' in vertices:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    else:
        colors = np.vstack([vertices['x'] * 0 + 180, vertices['y'] * 0 + 180, vertices['z'] * 0 + 180]).T / 255.0
    if 'nx' in vertices:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros_like(positions)
    if 'time' in vertices:
        timestamp = vertices['time'][:, None]
    else:
        timestamp = None
    return BasicPointCloud(points=positions, colors=colors, normals=normals, time=timestamp)

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L.transpose(1, 2) @ L
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.diffuse_activation = torch.sigmoid
        self.specular_activation = torch.tanh
        self.specular2_activation = torch.tanh
        self.roughness_activation = torch.sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, gaussian_dim : int = 3, time_duration: list = [-0.5, 0.5], rot_4d: bool = False, force_sh_3d: bool = False, sh_degree_t : int = 0, current_timestamp : float = 0.0, device: str = "cuda"):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0, device=device)
        self._features_dc = torch.empty(0, device=device)
        self._features_rest = torch.empty(0, device=device)
        self._scaling = torch.empty(0, device=device)
        self._rotation = torch.empty(0, device=device)
        self._opacity = torch.empty(0, device=device)
        self.max_radii2D = torch.empty(0, device=device)
        self.xyz_gradient_accum = torch.empty(0, device=device)
        self.xyz_gradient_accum_abs = torch.empty(0, device=device)
        self.specular_time_gradient_accum = torch.empty(0, device=device)
        self.denom = torch.empty(0, device=device)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        
        self.gaussian_dim = gaussian_dim
        self._t = torch.empty(0, device=device)
        self._scaling_t = torch.empty(0, device=device)
        self.time_duration = time_duration
        self.rot_4d = rot_4d
        self._velocity = torch.empty(0, device=device)
        self._velocity2 = torch.empty(0, device=device)
        self._velocity3 = torch.empty(0, device=device)
        self._rot_velocity = torch.empty(0, device = device)
        self.force_sh_3d = force_sh_3d
        self.t_gradient_accum = torch.empty(0, device=device)
        if self.rot_4d or self.force_sh_3d:
            assert self.gaussian_dim == 4
        self.env_map = torch.empty(0, device=device)
        
        self.active_sh_degree_t = 0
        self.max_sh_degree_t = sh_degree_t

        self.current_timestamp = current_timestamp
        self.setup_functions()
        self.opt_states = {}
        self.device = device
        #self.brdf_mlp = create_trainable_env_rnd(16, scale=0.0, bias=0.8)
        self.brdf_mlp = None
        #self.brdf_mlp_2 = create_trainable_env_rnd(128, scale=0.0, bias=0.8)
        self._specular = torch.empty(0, device=device)
        self._albedo = torch.empty(0, device=device)
        self._specular2 = torch.empty(0, device=device)
        self._roughness = torch.empty(0, device=device)
        self._delta_normal = torch.empty(0, device=device)
        self.default_roughness = 0.6
        self.temporal_opacity_mode = "normalized_sigmoid"
        self.temporal_flat_radius_mult = 0.75
        self.temporal_flat_edge_sigma_mult = 2.0
        self.temporal_flat_radius_start_mult = self.temporal_flat_radius_mult
        self.temporal_flat_radius_final_mult = self.temporal_flat_radius_mult
        self.temporal_flat_radius_ramp_start = 0
        self.temporal_flat_radius_ramp_end = 0
        self.temporal_flat_range_level = 0.05
        self.temporal_schedule_iteration = None

        
        # 16-dim env feature. This is the single knob that controls the feature
        # dim (init_light_env passes it explicitly into SphMipEncoding and derives
        # light_mlp/light_mlp_2 input widths from it). 32 was tried twice (exp76,
        # exp81) and finished 0.09-0.18 dB behind the sph16 twin both times.
        self.sph_dim = 16
        self.dim = 1
        self.gsdim = 4
        self.sph_time_keyframes = 1
        # self.sph_time_min = float(self.time_duration[0])
        # self.sph_time_max = float(self.time_duration[1])
        # self.sph_time_min = 1.6666666666666667
        # self.sph_time_max = 2.6333333333333333
        self.sph_time_min = 0.0
        self.sph_time_max = 1.9666666666666666
        self.sph_residual_keyframes = 0
        self.sph_residual_from_iter = 0
        self.sph_hierarchy_bands = ""
        self.sph_parity_bands = ""
        self.sph_sliding_bands = ""
        self.sph_sliding_window = 16
        # self.dir_encoding = SphMipEncoding(n_levels, plane_size, self.sph_dim, 1, self.dim, False).cuda()
        # self.light_mlp = nn.Sequential(
        #     nn.Linear(self.sph_dim * 4 + self.sph_dim, run_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(run_dim, run_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(run_dim, 3),
        # ).cuda()
        # nn.init.constant_(self.light_mlp[-1].bias, np.log(0.25))

        self.dir_encoding = None
        self.light_mlp = None
        self.light_mlp_2 = None

    def init_light_env(self):
        n_levels = 9  
        plane_size = 2**(n_levels)
        run_dim = 256
        self.dir_encoding = SphMipEncoding(
            n_levels,
            plane_size,
            self.sph_dim,
            self.sph_time_keyframes,
            self.dim,
            False,
            time_min=self.sph_time_min,
            time_max=self.sph_time_max,
            residual_keyframes=getattr(self, "sph_residual_keyframes", 0),
            residual_start_iteration=getattr(self, "sph_residual_from_iter", 0),
            residual_bands=getattr(self, "sph_hierarchy_bands", ""),
            parity_bands=getattr(self, "sph_parity_bands", ""),
            sliding_bands=getattr(self, "sph_sliding_bands", ""),
            sliding_window=getattr(self, "sph_sliding_window", 16),
        ).cuda()
        light_mlp_in_dim = self.sph_dim * self.gsdim + self.sph_dim
        parity_channels = self.dir_encoding.parity_channels
        if parity_channels > 0:
            # [spec_feat, f (x) s_static, f (x) (w_a * s_slotA), f (x) (w_b * s_slotB), w_a, w_b]
            light_mlp_in_dim += 2 * parity_channels * self.gsdim + 2
        sliding_slots = self.dir_encoding.sliding_slots
        if sliding_slots > 0:
            # [..., alpha_s * s_0..s_{W-1}, f (x) (alpha_s * s_0..s_{W-1}), alpha_0..alpha_{W-1}]
            sliding_dim = sliding_slots * self.dir_encoding.sliding_channels
            light_mlp_in_dim += sliding_dim + sliding_dim * self.gsdim + sliding_slots
        self.light_mlp = nn.Sequential(
            nn.Linear(light_mlp_in_dim, run_dim),
            nn.ReLU(inplace=True),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(inplace=True),
            nn.Linear(run_dim, 3),
        ).cuda()
        # self.light_mlp = nn.Sequential(
        #     nn.Linear(self.sph_dim * self.gsdim * 10 + self.sph_dim, run_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(run_dim, run_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(run_dim, 3),
        # ).cuda()
        # self.light_mlp = nn.Sequential(
        #     nn.Linear(self.sph_dim * self.gsdim + self.sph_dim + 10, run_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(run_dim, run_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(run_dim, 3),
        # ).cuda()
        # self.light_mlp = nn.Sequential(
        #     nn.Linear(self.sph_dim * 44 + self.sph_dim, run_dim * 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(run_dim * 2, run_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(run_dim, 3),
        # ).cuda()
        # self.light_mlp = nn.Sequential(
        #     nn.Linear(self.sph_dim * 44 + self.sph_dim, run_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(run_dim, run_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(run_dim, 3),
        # ).cuda()
        # self.light_mlp = nn.Sequential(
        #     nn.Linear(self.sph_dim * 48 + self.sph_dim, run_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(run_dim, run_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(run_dim, 3),
        # ).cuda()
        # self.light_mlp = nn.Sequential(
        #     nn.Linear(self.sph_dim * 114 + self.sph_dim, run_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(run_dim, run_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(run_dim, 3),
        # ).cuda()
        # self.light_mlp = nn.Sequential(
        #     nn.Linear(self.sph_dim * self.gsdim + self.sph_dim + 3, run_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(run_dim, run_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(run_dim, 3),
        # ).cuda()
        # self.light_mlp = nn.Sequential(
        #     nn.Linear(self.sph_dim * 4 + self.sph_dim, run_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(run_dim, run_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(run_dim, 3),
        # ).cuda()
        nn.init.constant_(self.light_mlp[-1].bias, np.log(0.25))
        # self.light_mlp_2 = nn.Sequential(
        #     nn.Linear(self.gsdim + 10, run_dim // 4),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(run_dim // 4, run_dim // 4),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(run_dim // 4, 4),
        # ).cuda()
        # nn.init.constant_(self.light_mlp_2[-1].bias, 0.0)
        # Single spec-light MLP: [global_feature, local_feature, roughness, cos(normal, reflect_dir)] -> RGB.
        # self.light_mlp_2 = nn.Sequential(
        #     nn.Linear(self.gsdim * self.gsdim + self.gsdim * self.sph_dim + self.sph_dim + 2, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 3),
        # ).cuda()
        # nn.init.constant_(self.light_mlp_2[-1].bias, np.log(0.25))

        # self.light_mlp_2 = nn.Sequential(
        #     nn.Linear(self.gsdim + 2, 64),
        #     nn.ReLU(inplace=True),
        #     # nn.Linear(64, 64),
        #     # nn.ReLU(inplace=True),
        #     nn.Linear(64, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 3),
        # ).cuda()
        self.light_mlp_2 = nn.Sequential(
            nn.Linear(self.gsdim + self.sph_dim + 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        ).cuda()
        nn.init.constant_(self.light_mlp_2[-1].bias, np.log(0.25))        

        # self.light_mlp_2 = SpecLightMLP(
        #     base_dim=self.gsdim + self.sph_dim + 2,
        #     outer_dim=self.sph_dim * self.gsdim,
        #     hidden_dim=128,
        #     out_dim=3,
        # ).cuda()
        # nn.init.constant_(self.light_mlp_2.fc_out.bias, np.log(0.25))
        self.brdf_mlp = create_trainable_env_rnd(16, scale=0.0, bias=0.8)

    def ensure_parity_light_env(self):
        """Idempotent setup/upgrade of the dynamic sph keyframe modes (parity
        slots and/or sliding window): creates the keyframe stacks on
        dir_encoding and widens light_mlp's first linear layer with zero-init
        columns for the new inputs (function-preserving for modules unpickled
        from older checkpoints, provided new input blocks are enabled in append
        order [parity, sliding]).  Must run before training_setup so the new
        params join the optimizer."""
        parity_bands = getattr(self, "sph_parity_bands", "")
        sliding_bands = getattr(self, "sph_sliding_bands", "")
        if (not parity_bands and not sliding_bands) or self.dir_encoding is None or self.light_mlp is None:
            return
        if parity_bands:
            self.dir_encoding.ensure_parity_keyframes(
                parity_bands,
                time_min=getattr(self, "sph_time_min", None),
                time_max=getattr(self, "sph_time_max", None),
            )
        if sliding_bands:
            self.dir_encoding.ensure_sliding_keyframes(
                sliding_bands,
                sliding_window=getattr(self, "sph_sliding_window", 16),
                time_min=getattr(self, "sph_time_min", None),
                time_max=getattr(self, "sph_time_max", None),
            )
        if not isinstance(self.light_mlp, nn.Sequential):
            return
        parity_channels = self.dir_encoding.parity_channels
        sliding_slots = self.dir_encoding.sliding_slots
        expected_in = self.sph_dim * self.gsdim + self.sph_dim
        if parity_channels > 0:
            expected_in += 2 * parity_channels * self.gsdim + 2
        sliding_dim = sliding_slots * self.dir_encoding.sliding_channels if sliding_slots > 0 else 0
        if sliding_slots > 0:
            expected_in += sliding_dim + sliding_dim * self.gsdim + sliding_slots
        first = self.light_mlp[0]
        if first.in_features == expected_in:
            return
        if first.in_features > expected_in:
            raise ValueError(f"[GaussianModel] light_mlp input {first.in_features} exceeds expected {expected_in}; "
                             f"checkpoint was trained with a larger dynamic sph spec.")
        # Zero-init widening is only function-preserving from a prefix of the
        # current layout: static-only, or static+parity when sliding is newly
        # added.  Any other width (legacy raw-concat sliding, a different
        # window/channel spec) would have its columns silently reinterpreted.
        valid_prefixes = {self.sph_dim * self.gsdim + self.sph_dim}
        if parity_channels > 0:
            valid_prefixes.add(self.sph_dim * self.gsdim + self.sph_dim + 2 * parity_channels * self.gsdim + 2)
        if first.in_features not in valid_prefixes:
            raise ValueError(f"[GaussianModel] light_mlp input {first.in_features} matches neither the current "
                             f"layout ({expected_in}) nor a function-preserving prefix ({sorted(valid_prefixes)}); "
                             f"the checkpoint used a different dynamic sph layout (e.g. legacy raw-concat sliding "
                             f"or another window/channel spec). Retrain from scratch.")
        new_first = nn.Linear(expected_in, first.out_features).to(first.weight.device, first.weight.dtype)
        with torch.no_grad():
            new_first.weight.zero_()
            new_first.weight[:, :first.in_features].copy_(first.weight)
            new_first.bias.copy_(first.bias)
        self.light_mlp[0] = new_first
        print(f"[GaussianModel] light_mlp input widened {first.in_features} -> {expected_in} "
              f"for dynamic sph keyframes (zero-init new columns).")

    def capture(self):
        if self.gaussian_dim == 3:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.xyz_gradient_accum_abs,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
        elif self.gaussian_dim == 4:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.xyz_gradient_accum_abs,
                self.t_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
                self._t,
                self._scaling_t,
                self._velocity,
                self._velocity2,
                self._velocity3,
                self._rot_velocity,
                self._specular,
                self._albedo,
                self._specular2,
                self._delta_normal,
                self._roughness,
                self.rot_4d,
                self.env_map,
                self.active_sh_degree_t,
            )
    
    def restore(self, model_args, training_args):
        if self.gaussian_dim == 3:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            xyz_gradient_accum_abs,
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        elif self.gaussian_dim == 4:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            xyz_gradient_accum_abs,
            t_gradient_accum,
            denom,
            opt_dict, 
            self.spatial_lr_scale,
            self._t,
            self._scaling_t,
            self._velocity,
            self._velocity2,
            self._velocity3,
            self._rot_velocity,
            self._specular,
            self._albedo,
            self._specular2,
            self._delta_normal,
            self._roughness,
            self.rot_4d,
            self.env_map,
            self.active_sh_degree_t,
            *_extra_args) = model_args

        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
            if self.gaussian_dim == 4:
                self.t_gradient_accum = t_gradient_accum
            self.denom = denom
            try:
                self.optimizer.load_state_dict(opt_dict)
            except Exception as exc:
                print(f"[GaussianModel] optimizer state restore skipped: {exc}")

    def clone_by_mask(self, mask : torch.Tensor, gaussians : "GaussianModel", opt, new_gaussians : "GaussianModel"):
       #new_gaussian = GaussianModel(self.sh_degree, self.gaussian_dim, self.time_duration, self.rot_4d, self.force_sh_3d, self.sh_degree_t)
        # new_gaussian.restore([self.active_sh_degree, self._xyz[mask], self._features_dc[mask], self._features_rest[mask], self._scaling[mask], self._rotation[mask],
        #                     self._opacity[mask], self.max_radii2D[mask], self.xyz_gradient_accum[mask], self.t_gradient_accum[mask], self.denom[mask],
        #                     self.spatial_lr_scale[mask], self._t[mask], self._scaling_t[mask], self._velocity[mask], self.rot_4d, self.env_map[mask], self.active_sh_degree_t], None)
        #new_gaussian.active_sh_degree = self.active_sh_degree
        if new_gaussians is None:
            self._xyz = gaussians._xyz[mask].to(self.device)
            self._features_dc = gaussians._features_dc[mask].to(self.device)
            self._features_rest = gaussians._features_rest[mask].to(self.device)
            self._scaling = gaussians._scaling[mask].to(self.device)
            self._rotation = gaussians._rotation[mask].to(self.device)
            self._opacity = gaussians._opacity[mask].to(self.device)
            self.max_radii2D = gaussians.max_radii2D[mask].to(self.device)
            self.xyz_gradient_accum = gaussians.xyz_gradient_accum[mask].to(self.device)
            self.xyz_gradient_accum_abs = gaussians.xyz_gradient_accum_abs[mask].to(self.device)
            self.t_gradient_accum = gaussians.t_gradient_accum[mask].to(self.device)
            self.denom = gaussians.denom[mask].to(self.device)
            #self.spatial_lr_scale = gaussians.spatial_lr_scale[mask].to(self.device)
            self._t = gaussians._t[mask].to(self.device)
            self._scaling_t = gaussians._scaling_t[mask].to(self.device)
            self._velocity = gaussians._velocity[mask].to(self.device)
            self._velocity2 = gaussians._velocity2[mask].to(self.device)
            self._velocity3 = gaussians._velocity3[mask].to(self.device)
            self._rot_velocity = gaussians._rot_velocity[mask].to(self.device)
            self._specular = gaussians._specular[mask].to(self.devices)
            self._albedo = gaussians._albedo[mask].to(self.devices)
            self._specular2 = gaussians._specular2[mask].to(self.devices)
            self._delta_normal = gaussians._delta_normal[mask].to(self.devices)
            self._roughness = gaussians._roughness[mask].to(self.devices)

            self.rot_4d = gaussians.rot_4d
            # if gaussians.env_map is not None:
            #     self.env_map = gaussians.env_map[mask].to(self.device)
            #new_gaussian.active_sh_degree_t = self.active_sh_degree_t
            #new_gaussian.percent_dense = self.percent_dense
            #new_gaussian.optimizer = self.optimizer
            #new_gaussian.max_sh_degree = self.max_sh_degree
            self.gaussian_dim = gaussians.gaussian_dim
            self.time_duration = gaussians.time_duration
            self.force_sh_3d = gaussians.force_sh_3d
            self.max_sh_degree_t = gaussians.max_sh_degree_t
            #self.training_setup(opt)
            #new_gaussian.setup_functions()
            self.opt_states.clear()
            for group in gaussians.optimizer.param_groups:
                optimizer_state = gaussians.optimizer.state.get(group["params"][0], None)
                #attr = self.get_param_group_corresponding_attr(group["name"])
                if optimizer_state is not None and len(optimizer_state) != 0:
                    state_content = {}
                    state_content["step"] = optimizer_state["step"].to(self.device)
                    state_content["exp_avg"] = optimizer_state["exp_avg"][mask].to(self.device)
                    state_content["exp_avg_sq"] = optimizer_state["exp_avg_sq"][mask].to(self.device)
                    self.opt_states[group["name"]] = state_content
        else:

            new_gaussians._xyz = torch.cat([new_gaussians._xyz, gaussians._xyz[mask].to(self.device)])
            new_gaussians._features_dc = torch.cat([new_gaussians._features_dc, gaussians._features_dc[mask].to(self.device)])
            new_gaussians._features_rest = torch.cat([new_gaussians._features_rest, gaussians._features_rest[mask].to(self.device)])
            new_gaussians._scaling = torch.cat([new_gaussians._scaling, gaussians._scaling[mask].to(self.device)])
            new_gaussians._rotation = torch.cat([new_gaussians._rotation, gaussians._rotation[mask].to(self.device)])
            new_gaussians._opacity = torch.cat([new_gaussians._opacity, gaussians._opacity[mask].to(self.device)])
            new_gaussians.max_radii2D = torch.cat([new_gaussians.max_radii2D, gaussians.max_radii2D[mask].to(self.device)])
            new_gaussians.xyz_gradient_accum = torch.cat([new_gaussians.xyz_gradient_accum, gaussians.xyz_gradient_accum[mask].to(self.device)])
            new_gaussians.xyz_gradient_accum_abs = torch.cat([new_gaussians.xyz_gradient_accum_abs, gaussians.xyz_gradient_accum_abs[mask].to(self.device)])
            new_gaussians.t_gradient_accum = torch.cat([new_gaussians.t_gradient_accum, gaussians.t_gradient_accum[mask].to(self.device)])
            new_gaussians.denom = torch.cat([new_gaussians.denom, gaussians.denom[mask].to(self.device)])
            #self.spatial_lr_scale = torch.cat([self.spatial_lr_scale, gaussians.spatial_lr_scale[mask]]).to(self.device)
            new_gaussians._t = torch.cat([new_gaussians._t, gaussians._t[mask].to(self.device)])
            new_gaussians._scaling_t = torch.cat([new_gaussians._scaling_t, gaussians._scaling_t[mask].to(self.device)])
            new_gaussians._velocity = torch.cat([new_gaussians._velocity, gaussians._velocity[mask].to(self.device)])
            new_gaussians._velocity2 = torch.cat([new_gaussians._velocity2, gaussians._velocity2[mask].to(self.device)])
            new_gaussians._velocity3 = torch.cat([new_gaussians._velocity3, gaussians._velocity3[mask].to(self.device)])
            new_gaussians._rot_velocity = torch.cat([new_gaussians._rot_velocity, gaussians._rot_velocity[mask].to(self.device)])
            new_gaussians._specular = torch.cat([new_gaussians._specular, gaussians._specular[mask].to(self.device)])
            new_gaussians._albedo = torch.cat([new_gaussians._albedo, gaussians._albedo[mask].to(self.device)])
            new_gaussians._specular2 = torch.cat([new_gaussians._specular2, gaussians._specular2[mask].to(self.device)])
            new_gaussians._delta_normal = torch.cat([new_gaussians._delta_normal, gaussians._delta_normal[mask].to(self.device)])

            # new_gaussians._xyz = gaussians._xyz[mask].to(self.device)
            # new_gaussians._features_dc = gaussians._features_dc[mask].to(self.device)
            # new_gaussians._features_rest = gaussians._features_rest[mask].to(self.device)
            # new_gaussians._scaling = gaussians._scaling[mask].to(self.device)
            # new_gaussians._rotation = gaussians._rotation[mask].to(self.device)
            # new_gaussians._opacity = gaussians._opacity[mask].to(self.device)
            # new_gaussians.max_radii2D = gaussians.max_radii2D[mask].to(self.device)
            # new_gaussians.xyz_gradient_accum = gaussians.xyz_gradient_accum[mask].to(self.device)
            # new_gaussians.t_gradient_accum = gaussians.t_gradient_accum[mask].to(self.device)
            # new_gaussians.denom = gaussians.denom[mask].to(self.device)
            # #self.spatial_lr_scale = gaussians.spatial_lr_scale[mask].to(self.device)
            # new_gaussians._t = gaussians._t[mask].to(self.device)
            # new_gaussians._scaling_t = gaussians._scaling_t[mask].to(self.device)
            # new_gaussians._velocity = gaussians._velocity[mask].to(self.device)

            # new_gaussians.rot_4d = gaussians.rot_4d
            # if gaussians.env_map is not None:
            #     self.env_map = gaussians.env_map[mask].to(self.device)
            #new_gaussian.active_sh_degree_t = self.active_sh_degree_t
            #new_gaussian.percent_dense = self.percent_dense
            #new_gaussian.optimizer = self.optimizer
            #new_gaussian.max_sh_degree = self.max_sh_degree
            # new_gaussians.gaussian_dim = gaussians.gaussian_dim
            # new_gaussians.time_duration = gaussians.time_duration
            # new_gaussians.force_sh_3d = gaussians.force_sh_3d
            # new_gaussians.max_sh_degree_t = gaussians.max_sh_degree_t
            # for group in gaussians.optimizer.param_groups:
            #     optimizer_state = gaussians.optimizer.state.get(group["params"][0], None)
            #     #attr = self.get_param_group_corresponding_attr(group["name"])
            #     if optimizer_state is not None and len(optimizer_state) != 0:
            #         state_content = {}
            #         state_content["step"] = optimizer_state["step"].to(self.device)
            #         state_content["exp_avg"] = optimizer_state["exp_avg"][mask].to(self.device)
            #         state_content["exp_avg_sq"] = optimizer_state["exp_avg_sq"][mask].to(self.device)
            #         new_gaussians.opt_states[group["name"]] = state_content

    def clone_from_cpu(self, gaussians_segments):
        xyz_list = []
        for segment in gaussians_segments:
            xyz_list.append(segment._xyz.cuda())
        #del self.optimizer.state[self._xyz]
        self._xyz = nn.Parameter(torch.cat(xyz_list))

        features_dc_list = []
        for segment in gaussians_segments:
            features_dc_list.append(segment._features_dc.cuda())
        self._features_dc = nn.Parameter(torch.cat(features_dc_list))

        features_rest_list = []
        for segment in gaussians_segments:
            features_rest_list.append(segment._features_rest.cuda())
        self._features_rest = nn.Parameter(torch.cat(features_rest_list))

        scaling_list = []
        for segment in gaussians_segments:
            scaling_list.append(segment._scaling.cuda())
        self._scaling = nn.Parameter(torch.cat(scaling_list))

        rotation_list = []
        for segment in gaussians_segments:
            rotation_list.append(segment._rotation.cuda())
        self._rotation = nn.Parameter(torch.cat(rotation_list))

        opacity_list = []
        for segment in gaussians_segments:
            opacity_list.append(segment._opacity.cuda())
        self._opacity = nn.Parameter(torch.cat(opacity_list))

        t_list = []
        for segment in gaussians_segments:
            t_list.append(segment._t.cuda())
        self._t = nn.Parameter(torch.cat(t_list))

        scaling_t_list = []
        for segment in gaussians_segments:
            scaling_t_list.append(segment._scaling_t.cuda())
        self._scaling_t = nn.Parameter(torch.cat(scaling_t_list))
        # print("clone_from_cpu")
        # print(self._scaling_t)

        velocity_list = []
        for segment in gaussians_segments:
            velocity_list.append(segment._velocity.cuda())
        self._velocity = nn.Parameter(torch.cat(velocity_list))

        velocity2_list = []
        for segment in gaussians_segments:
            velocity2_list.append(segment._velocity2.cuda())
        self._velocity2 = nn.Parameter(torch.cat(velocity2_list))

        velocity3_list = []
        for segment in gaussians_segments:
            velocity3_list.append(segment._velocity3.cuda())
        self._velocity3 = nn.Parameter(torch.cat(velocity3_list))

        rot_velocity_list = []
        for segment in gaussians_segments:
            rot_velocity_list.append(segment._rot_velocity.cuda())
        self._rot_velocity = nn.Parameter(torch.cat(rot_velocity_list))

        specular_list = []
        for segment in gaussians_segments:
            specular_list.append(segment._specular.cuda())
        self._specular = nn.Parameter(torch.cat(specular_list))

        albedo_list = []
        for segment in gaussians_segments:
            albedo_list.append(segment._albedo.cuda())
        self._albedo = nn.Parameter(torch.cat(albedo_list))

        specular2_list = []
        for segment in gaussians_segments:
            specular2_list.append(segment._specular2.cuda())
        self._specular2 = nn.Parameter(torch.cat(specular2_list))

        delta_normal_list = []
        for segment in gaussians_segments:
            delta_normal_list.append(segment._delta_normal.cuda())
        self._delta_normal = nn.Parameter(torch.cat(delta_normal_list))

        roughness_list = []
        for segment in gaussians_segments:
            roughness_list.append(segment._roughness.cuda())
        self._roughness = nn.Parameter(torch.cat(roughness_list))

        max_radii2D_list = []
        for segment in gaussians_segments:
            max_radii2D_list.append(segment.max_radii2D.cuda())
        self.max_radii2D = torch.cat(max_radii2D_list)

        xyz_gradient_accum_list = []
        for segment in gaussians_segments:
            xyz_gradient_accum_list.append(segment.xyz_gradient_accum.cuda())
        self.xyz_gradient_accum = torch.cat(xyz_gradient_accum_list)

        xyz_gradient_accum_abs_list = []
        for segment in gaussians_segments:
            xyz_gradient_accum_abs_list.append(segment.xyz_gradient_accum_abs.cuda())
        self.xyz_gradient_accum_abs = torch.cat(xyz_gradient_accum_abs_list)

        t_gradient_accum_list = []
        for segment in gaussians_segments:
            t_gradient_accum_list.append(segment.t_gradient_accum.cuda())
        self.t_gradient_accum = torch.cat(t_gradient_accum_list)

        denom_list = []
        for segment in gaussians_segments:
            denom_list.append(segment.denom.cuda())
        self.denom = torch.cat(denom_list)

        self.rot_4d = gaussians_segments[0].rot_4d
        self.gaussian_dim = gaussians_segments[0].gaussian_dim
        self.time_duration = gaussians_segments[0].time_duration
        self.force_sh_3d = gaussians_segments[0].force_sh_3d
        self.max_sh_degree_t = gaussians_segments[0].max_sh_degree_t
    
    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = quaternion_to_matrix(self.get_rotation)
        smallest_axis_idx = self.get_scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)

    def get_normal(self, camera_center, xyz):
        normal_global = self.get_smallest_axis()
        # normal_global = rotation
        gaussian_to_cam_global = camera_center - xyz
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global

    def clear_gaussians(self, gaussians):
        self._xyz = torch.empty(0, 3, device=self.device)
        self._features_dc = torch.empty(0, 1, 3, device=self.device)
        self._features_rest = torch.empty(0, self.get_max_sh_channels - 1, 3, device=self.device)
        self._scaling = torch.empty(0, 3, device=self.device)
        self._rotation = torch.empty(0, 4, device=self.device)
        self._opacity = torch.empty(0, 1, device=self.device)
        self.max_radii2D = torch.empty(0, device=self.device)
        self.xyz_gradient_accum = torch.empty(0, 1, device=self.device)
        self.xyz_gradient_accum_abs = torch.empty(0, 1, device=self.device)
        self.t_gradient_accum = torch.empty(0, 1, device=self.device)
        self.denom = torch.empty(0, 1, device=self.device)
        self._t = torch.empty(0, 1, device=self.device)
        self._scaling_t = torch.empty(0, 1, device=self.device)
        self._velocity = torch.empty(0, 3, device=self.device)
        self._velocity2 = torch.empty(0, 3, device=self.device)
        self._velocity3 = torch.empty(0, 3, device=self.device)
        self._rot_velocity = torch.empty(0, 4, device=self.device)
        self._specular = torch.empty(0, 4, device=self.device)
        self._albedo = torch.empty(0, 3, device=self.device)
        self._specular2 = torch.empty(0, 44, device=self.device)
        #self._specular2 = torch.empty(0, 20, device=self.device)
        #self._specular2 = torch.empty(0, 4, device=self.device)
        self._delta_normal = torch.empty(0, 3, device=self.device)
        self._roughness = torch.empty(0, 1, device=self.device)

        self.opt_states.clear()
        if gaussians is None:
            return
        self.rot_4d = gaussians.rot_4d
        self.gaussian_dim = gaussians.gaussian_dim
        self.time_duration = gaussians.time_duration
        self.force_sh_3d = gaussians.force_sh_3d
        self.max_sh_degree_t = gaussians.max_sh_degree_t
        #self.training_setup(opt)
        #new_gaussian.setup_functions()
        for group in gaussians.optimizer.param_groups:
            optimizer_state = gaussians.optimizer.state.get(group["params"][0], None)
            #attr = self.get_param_group_corresponding_attr(group["name"])
            if optimizer_state is not None and len(optimizer_state) != 0:
                state_content = {}
                state_content["step"] = optimizer_state["step"].to(self.device)
                state_content["exp_avg"] = torch.empty(0, device=self.device)
                state_content["exp_avg_sq"] = torch.empty(0, device=self.device)
                self.opt_states[group["name"]] = state_content
    
    # def clone_to(self, gaussians : "GaussianModel"):
    #     #new_gaussian = GaussianModel(self.sh_degree, self.gaussian_dim, self.time_duration, self.rot_4d, self.force_sh_3d, self.sh_degree_t)
    #     # new_gaussian.restore([self.active_sh_degree, self._xyz, self._features_dc, self._features_rest, self._scaling, self._rotation,
    #     #                     self._opacity, self.max_radii2D, self.xyz_gradient_accum, self.t_gradient_accum, self.denom,
    #     #                     self.spatial_lr_scale, self._t, self._scaling_t, self._velocity, self.rot_4d, self.env_map, self.active_sh_degree_t], None)
    #     #gaussians.active_sh_degree = self.active_sh_degree
    #     state_dict = {}
    #     if gaussians._xyz in gaussians.optimizer.state:
    #         state_dict["xyz"] = gaussians.optimizer.state[gaussians._xyz]
    #         del gaussians.optimizer.state[gaussians._xyz]
    #     gaussians._xyz = nn.Parameter(self._xyz.cuda())

    #     if gaussians._features_dc in gaussians.optimizer.state:
    #         state_dict["f_dc"] = gaussians.optimizer.state[gaussians._features_dc]
    #         del gaussians.optimizer.state[gaussians._features_dc]
    #     gaussians._features_dc = nn.Parameter(self._features_dc.cuda())

    #     if gaussians._features_rest in gaussians.optimizer.state:
    #         state_dict["f_rest"] = gaussians.optimizer.state[gaussians._features_rest]
    #         del gaussians.optimizer.state[gaussians._features_rest]
    #     gaussians._features_rest = nn.Parameter(self._features_rest.cuda())

    #     if gaussians._scaling in gaussians.optimizer.state:
    #         state_dict["scaling"] = gaussians.optimizer.state[gaussians._scaling]
    #         del gaussians.optimizer.state[gaussians._scaling]
    #     gaussians._scaling = nn.Parameter(self._scaling.cuda())

    #     if gaussians._rotation in gaussians.optimizer.state:
    #         state_dict["rotation"] = gaussians.optimizer.state[gaussians._rotation]
    #         del gaussians.optimizer.state[gaussians._rotation]
    #     gaussians._rotation = nn.Parameter(self._rotation.cuda())

    #     if gaussians._opacity in gaussians.optimizer.state:
    #         state_dict["opacity"] = gaussians.optimizer.state[gaussians._opacity]
    #         del gaussians.optimizer.state[gaussians._opacity]
    #     gaussians._opacity = nn.Parameter(self._opacity.cuda())

    #     gaussians.max_radii2D = self.max_radii2D.cuda()

    #     gaussians.xyz_gradient_accum = self.xyz_gradient_accum.cuda()
    #     gaussians.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs.cuda()
    #     gaussians.t_gradient_accum = self.t_gradient_accum.cuda()
    #     gaussians.denom = self.denom.cuda()
    #     #gaussians.spatial_lr_scale = self.spatial_lr_scale.cuda()
    #     if gaussians._t in gaussians.optimizer.state:
    #         state_dict["t"] = gaussians.optimizer.state[gaussians._t]
    #         del gaussians.optimizer.state[gaussians._t]
    #     gaussians._t = nn.Parameter(self._t.cuda())

    #     if gaussians._scaling_t in gaussians.optimizer.state:
    #         state_dict["scaling_t"] = gaussians.optimizer.state[gaussians._scaling_t]
    #         del gaussians.optimizer.state[gaussians._scaling_t]
    #     gaussians._scaling_t = nn.Parameter(self._scaling_t.cuda())

    #     if gaussians._velocity in gaussians.optimizer.state:
    #         state_dict["velocity"] = gaussians.optimizer.state[gaussians._velocity]
    #         del gaussians.optimizer.state[gaussians._velocity]
    #     gaussians._velocity = nn.Parameter(self._velocity.cuda())

    #     gaussians.rot_4d = self.rot_4d
    #     # if self.env_map is not None:
    #     #     gaussians.env_map = self.env_map.cuda()
    #     #gaussians.active_sh_degree_t = self.active_sh_degree_t
    #     #gaussians.percent_dense = self.percent_dense
    #     #new_gaussian.optimizer = self.optimizer
    #     #gaussians.max_sh_degree = self.max_sh_degree
    #     gaussians.gaussian_dim = self.gaussian_dim
    #     gaussians.time_duration = self.time_duration
    #     gaussians.force_sh_3d = self.force_sh_3d
    #     gaussians.max_sh_degree_t = self.max_sh_degree_t
    #     return state_dict
    
    def get_state_dict(self):
        #new_gaussian = GaussianModel(self.sh_degree, self.gaussian_dim, self.time_duration, self.rot_4d, self.force_sh_3d, self.sh_degree_t)
        # new_gaussian.restore([self.active_sh_degree, self._xyz, self._features_dc, self._features_rest, self._scaling, self._rotation,
        #                     self._opacity, self.max_radii2D, self.xyz_gradient_accum, self.t_gradient_accum, self.denom,
        #                     self.spatial_lr_scale, self._t, self._scaling_t, self._velocity, self.rot_4d, self.env_map, self.active_sh_degree_t], None)
        #gaussians.active_sh_degree = self.active_sh_degree
        state_dict = {}
        if self._xyz in self.optimizer.state:
            state_dict["xyz"] = self.optimizer.state[self._xyz]
            del self.optimizer.state[self._xyz]
        #gaussians._xyz = nn.Parameter(self._xyz.cuda())

        if self._features_dc in self.optimizer.state:
            state_dict["f_dc"] = self.optimizer.state[self._features_dc]
            del self.optimizer.state[self._features_dc]
        #gaussians._features_dc = nn.Parameter(self._features_dc.cuda())

        if self._features_rest in self.optimizer.state:
            state_dict["f_rest"] = self.optimizer.state[self._features_rest]
            del self.optimizer.state[self._features_rest]
        #gaussians._features_rest = nn.Parameter(self._features_rest.cuda())

        if self._scaling in self.optimizer.state:
            state_dict["scaling"] = self.optimizer.state[self._scaling]
            del self.optimizer.state[self._scaling]
        #gaussians._scaling = nn.Parameter(self._scaling.cuda())

        if self._rotation in self.optimizer.state:
            state_dict["rotation"] = self.optimizer.state[self._rotation]
            del self.optimizer.state[self._rotation]
        #gaussians._rotation = nn.Parameter(self._rotation.cuda())

        if self._opacity in self.optimizer.state:
            state_dict["opacity"] = self.optimizer.state[self._opacity]
            del self.optimizer.state[self._opacity]
        #gaussians._opacity = nn.Parameter(self._opacity.cuda())

        #gaussians.max_radii2D = self.max_radii2D.cuda()

        #gaussians.xyz_gradient_accum = self.xyz_gradient_accum.cuda()
        #gaussians.t_gradient_accum = self.t_gradient_accum.cuda()
        #gaussians.denom = self.denom.cuda()
        #gaussians.spatial_lr_scale = self.spatial_lr_scale.cuda()
        if self._t in self.optimizer.state:
            state_dict["t"] = self.optimizer.state[self._t]
            del self.optimizer.state[self._t]
        #gaussians._t = nn.Parameter(self._t.cuda())

        if self._scaling_t in self.optimizer.state:
            state_dict["scaling_t"] = self.optimizer.state[self._scaling_t]
            del self.optimizer.state[self._scaling_t]
        #gaussians._scaling_t = nn.Parameter(self._scaling_t.cuda())

        if self._velocity in self.optimizer.state:
            state_dict["velocity"] = self.optimizer.state[self._velocity]
            del self.optimizer.state[self._velocity]
        
        if self._velocity2 in self.optimizer.state:
            state_dict["velocity2"] = self.optimizer.state[self._velocity2]
            del self.optimizer.state[self._velocity2]

        if self._velocity3 in self.optimizer.state:
            state_dict["velocity3"] = self.optimizer.state[self._velocity3]
            del self.optimizer.state[self._velocity3]
        
        if self._rot_velocity in self.optimizer.state:
            state_dict["rot_velocity"] = self.optimizer.state[self._rot_velocity]
            del self.optimizer.state[self._rot_velocity]

        if self._specular in self.optimizer.state:
            state_dict["specular"] = self.optimizer.state[self._specular]
            del self.optimizer.state[self._specular]

        if self._albedo in self.optimizer.state:
            state_dict["albedo"] = self.optimizer.state[self._albedo]
            del self.optimizer.state[self._albedo]

        if self._specular2 in self.optimizer.state:
            state_dict["specular2"] = self.optimizer.state[self._specular2]
            del self.optimizer.state[self._specular2]

        if self._delta_normal in self.optimizer.state:
            state_dict["delta_normal"] = self.optimizer.state[self._delta_normal]
            del self.optimizer.state[self._delta_normal]

        if self._roughness in self.optimizer.state:
            state_dict["roughness"] = self.optimizer.state[self._roughness]
            del self.optimizer.state[self._roughness]
        #gaussians._velocity = nn.Parameter(self._velocity.cuda())

        #gaussians.rot_4d = self.rot_4d
        # if self.env_map is not None:
        #     gaussians.env_map = self.env_map.cuda()
        #gaussians.active_sh_degree_t = self.active_sh_degree_t
        #gaussians.percent_dense = self.percent_dense
        #new_gaussian.optimizer = self.optimizer
        #gaussians.max_sh_degree = self.max_sh_degree
        #gaussians.gaussian_dim = self.gaussian_dim
        #gaussians.time_duration = self.time_duration
        #gaussians.force_sh_3d = self.force_sh_3d
        #gaussians.max_sh_degree_t = self.max_sh_degree_t
        return state_dict
        

    def reset_param_groups(self):
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                group["params"][0] = self._xyz
            if group["name"] == "f_dc":
                group["params"][0] = self._features_dc
            if group["name"] == "f_rest":
                group["params"][0] = self._features_rest
            if group["name"] == "opacity":
                group["params"][0] = self._opacity
            if group["name"] == "scaling":
                group["params"][0] = self._scaling
            if group["name"] == "rotation":
                group["params"][0] = self._rotation
            if group["name"] == "t":
                group["params"][0] = self._t
            if group["name"] == "scaling_t":
                group["params"][0] = self._scaling_t
            if group["name"] == "velocity":
                group["params"][0] = self._velocity
            if group["name"] == "velocity2":
                group["params"][0] = self._velocity2
            if group["name"] == "velocity3":
                group["params"][0] = self._velocity3
            if group["name"] == "rot_velocity":
                group["params"][0] = self._rot_velocity
            if group["name"] == "specular":
                group["params"][0] = self._specular
            if group["name"] == "albedo":
                group["params"][0] = self._albedo
            if group["name"] == "specular2":
                group["params"][0] = self._specular2
            if group["name"] == "delta_normal":
                group["params"][0] = self._delta_normal
            if group["name"] == "roughness":
                group["params"][0] = self._roughness

    def get_param_group_corresponding_attr(self, group_name):
        if group_name == "xyz":
            return self._xyz
        if group_name == "f_dc":
            return self._features_dc
        if group_name == "f_rest":
            return self._features_rest
        if group_name == "opacity":
            return self._opacity
        if group_name == "scaling":
            return self._scaling
        if group_name == "rotation":
            return self._rotation
        if group_name == "t":
            return self._t
        if group_name == "scaling_t":
            return self._scaling_t
        if group_name == "velocity":
            return self._velocity
        if group_name == "velocity2":
            return self._velocity2
        if group_name == "velocity3":
            return self._velocity3
        if group_name == "rot_velocity":
            return self._rot_velocity
        if group_name == "specular":
            return self._specular
        if group_name == "albedo":
            return self._albedo
        if group_name == "specular2":
            return self._specular2
        if group_name == "delta_normal":
            return self._delta_normal
        if group_name == "roughness":
            return self._roughness
        return None

    def append_from_gaussians_gpu(self, mask : torch.Tensor, gaussians : "GaussianModel", new_gaussians : "GaussianModel"):
        if new_gaussians is None:
            self._xyz = torch.cat([self._xyz, gaussians._xyz[mask].to(self.device)])
            self._features_dc = torch.cat([self._features_dc, gaussians._features_dc[mask].to(self.device)])
            self._features_rest = torch.cat([self._features_rest, gaussians._features_rest[mask].to(self.device)])
            self._scaling = torch.cat([self._scaling, gaussians._scaling[mask].to(self.device)])
            self._rotation = torch.cat([self._rotation, gaussians._rotation[mask].to(self.device)])
            self._opacity = torch.cat([self._opacity, gaussians._opacity[mask].to(self.device)])
            self.max_radii2D = torch.cat([self.max_radii2D, gaussians.max_radii2D[mask].to(self.device)])
            self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, gaussians.xyz_gradient_accum[mask].to(self.device)])
            self.xyz_gradient_accum_abs = torch.cat([self.xyz_gradient_accum_abs, gaussians.xyz_gradient_accum_abs[mask].to(self.device)])
            self.specular_time_gradient_accum = torch.cat([self.specular_time_gradient_accum, gaussians.specular_time_gradient_accum[mask].to(self.device)])
            self.t_gradient_accum = torch.cat([self.t_gradient_accum, gaussians.t_gradient_accum[mask].to(self.device)])
            self.denom = torch.cat([self.denom, gaussians.denom[mask].to(self.device)])
            #self.spatial_lr_scale = torch.cat([self.spatial_lr_scale, gaussians.spatial_lr_scale[mask]]).to(self.device)
            self._t = torch.cat([self._t, gaussians._t[mask].to(self.device)])
            self._scaling_t = torch.cat([self._scaling_t, gaussians._scaling_t[mask].to(self.device)])
            # print(self._velocity.device, gaussians._velocity.device)
            self._velocity = torch.cat([self._velocity, gaussians._velocity[mask].to(self.device)])
            self._velocity2 = torch.cat([self._velocity2, gaussians._velocity2[mask].to(self.device)])
            self._velocity3 = torch.cat([self._velocity3, gaussians._velocity3[mask].to(self.device)])
            self._rot_velocity = torch.cat([self._rot_velocity, gaussians._rot_velocity[mask].to(self.device)])
            self._specular = torch.cat([self._specular, gaussians._specular[mask].to(self.device)])
            self._albedo = torch.cat([self._albedo, gaussians._albedo[mask].to(self.device)])
            self._specular2 = torch.cat([self._specular2, gaussians._specular2[mask].to(self.device)])
            self._delta_normal = torch.cat([self._delta_normal, gaussians._delta_normal[mask].to(self.device)])
            self._roughness = torch.cat([self._roughness, gaussians._roughness[mask].to(self.device)])
            # if gaussians.env_map is not None:
            #     self.env_map = torch.cat([self.env_map, gaussians.env_map[mask]]).to(self.device)
            for group in gaussians.optimizer.param_groups:
                if group["name"] == "brdf_mlp":
                    continue
                if group["name"] == "light_mlp":
                    continue
                if group["name"] == "light_mlp2":
                    continue
                if group["name"] == "dir_encoding":
                    continue
                optimizer_state = gaussians.optimizer.state.get(group["params"][0], None)
                if optimizer_state is not None:
                    state_content = {}
                    state_content["step"] = optimizer_state["step"].to(self.device)
                    state_content["exp_avg"] = optimizer_state["exp_avg"][mask].to(self.device)
                    state_content["exp_avg_sq"] = optimizer_state["exp_avg_sq"][mask].to(self.device)
                    #attr = self.get_param_group_corresponding_attr(group["name"])
                    tgh_gaussian_state = self.opt_states.get(group["name"], None)
                    if tgh_gaussian_state is not None:
                        tgh_gaussian_state["step"] = state_content["step"]
                        tgh_gaussian_state["exp_avg"] = torch.cat([tgh_gaussian_state["exp_avg"], state_content["exp_avg"]])
                        tgh_gaussian_state["exp_avg_sq"] = torch.cat([tgh_gaussian_state["exp_avg_sq"], state_content["exp_avg_sq"]])
                        #self.opt_states[group["name"]] = tgh_gaussian_state
                    else:
                        self.opt_states[group["name"]] = state_content
                else:
                    #pass
                    if group["name"] == "velocity2" or group["name"] == "velocity3" or group["name"] == "rot_velocity" or group["name"] == "brdf_mlp" or group["name"] == "f_rest":
                        continue
                    self.opt_states.update({group["name"]: {"step": torch.tensor(1., device=self.device), "exp_avg": torch.zeros_like(self.get_param_group_corresponding_attr(group["name"])), "exp_avg_sq": torch.zeros_like(self.get_param_group_corresponding_attr(group["name"]))}})
        else:
            new_gaussians._xyz = torch.cat([new_gaussians._xyz, self._xyz, gaussians._xyz[mask].to(self.device)])
            new_gaussians._features_dc = torch.cat([new_gaussians._features_dc, self._features_dc, gaussians._features_dc[mask].to(self.device)])
            new_gaussians._features_rest = torch.cat([new_gaussians._features_rest, self._features_rest, gaussians._features_rest[mask].to(self.device)])
            new_gaussians._scaling = torch.cat([new_gaussians._scaling, self._scaling, gaussians._scaling[mask].to(self.device)])
            new_gaussians._rotation = torch.cat([new_gaussians._rotation, self._rotation, gaussians._rotation[mask].to(self.device)])
            new_gaussians._opacity = torch.cat([new_gaussians._opacity, self._opacity, gaussians._opacity[mask].to(self.device)])
            new_gaussians.max_radii2D = torch.cat([new_gaussians.max_radii2D, self.max_radii2D, gaussians.max_radii2D[mask].to(self.device)])
            new_gaussians.xyz_gradient_accum = torch.cat([new_gaussians.xyz_gradient_accum, self.xyz_gradient_accum, gaussians.xyz_gradient_accum[mask].to(self.device)])
            new_gaussians.xyz_gradient_accum_abs = torch.cat([new_gaussians.xyz_gradient_accum_abs, self.xyz_gradient_accum_abs, gaussians.xyz_gradient_accum_abs[mask].to(self.device)])
            new_gaussians.t_gradient_accum = torch.cat([new_gaussians.t_gradient_accum, self.t_gradient_accum, gaussians.t_gradient_accum[mask].to(self.device)])
            new_gaussians.denom = torch.cat([new_gaussians.denom, self.denom, gaussians.denom[mask].to(self.device)])
            #self.spatial_lr_scale = torch.cat([self.spatial_lr_scale, gaussians.spatial_lr_scale[mask]]).to(self.device)
            new_gaussians._t = torch.cat([new_gaussians._t, self._t, gaussians._t[mask].to(self.device)])
            new_gaussians._scaling_t = torch.cat([new_gaussians._scaling_t, self._scaling_t, gaussians._scaling_t[mask].to(self.device)])
            new_gaussians._velocity = torch.cat([new_gaussians._velocity, self._velocity, gaussians._velocity[mask].to(self.device)])
            new_gaussians._velocity2 = torch.cat([new_gaussians._velocity2, self._velocity2, gaussians._velocity2[mask].to(self.device)])
            new_gaussians._velocity3 = torch.cat([new_gaussians._velocity3, self._velocity3, gaussians._velocity3[mask].to(self.device)])
            new_gaussians._rot_velocity = torch.cat([new_gaussians._rot_velocity, self._rot_velocity, gaussians._rot_velocity[mask].to(self.device)])
            new_gaussians._specular = torch.cat([new_gaussians._specular, self._specular, gaussians._specular[mask].to(self.device)])
            new_gaussians._albedo = torch.cat([new_gaussians._albedo, self._albedo, gaussians._albedo[mask].to(self.device)])
            new_gaussians._specular2 = torch.cat([new_gaussians._specular2, self._specular2, gaussians._specular2[mask].to(self.device)])
            new_gaussians._delta_normal = torch.cat([new_gaussians._delta_normal, self._delta_normal, gaussians._delta_normal[mask].to(self.device)])
            new_gaussians._roughness = torch.cat([new_gaussians._roughness, self._roughness, gaussians._roughness[mask].to(self.device)])
            # if gaussians.env_map is not None:
            #     self.env_map = torch.cat([self.env_map, gaussians.env_map[mask]]).to(self.device)
            # for group in gaussians.optimizer.param_groups:
            #     optimizer_state = gaussians.optimizer.state.get(group["params"][0], None)
            #     if optimizer_state is not None:
            #         state_content = {}
            #         state_content["step"] = optimizer_state["step"].to(self.device)
            #         state_content["exp_avg"] = optimizer_state["exp_avg"][mask].to(self.device)
            #         state_content["exp_avg_sq"] = optimizer_state["exp_avg_sq"][mask].to(self.device)
            #         #attr = self.get_param_group_corresponding_attr(group["name"])
            #         tgh_gaussian_state = self.opt_states.get(group["name"], None)
            #         if tgh_gaussian_state is not None:
            #             state_content["exp_avg"] = torch.cat([new_gaussians.opt_states.get(group["name"])["exp_avg"], tgh_gaussian_state["exp_avg"], state_content["exp_avg"]])
            #             state_content["exp_avg_sq"] = torch.cat([new_gaussians.opt_states.get(group["name"])["exp_avg_sq"], tgh_gaussian_state["exp_avg_sq"], state_content["exp_avg_sq"]])
            #             #self.opt_states[group["name"]] = tgh_gaussian_state
            #         new_gaussians.opt_states[group["name"]] = state_content

        

    def append_from_gaussians_cpu(self, gaussians : "GaussianModel"):
        self._xyz = nn.Parameter(torch.cat([self._xyz, gaussians._xyz.cuda()]))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, gaussians._features_dc.cuda()]))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest, gaussians._features_rest.cuda()]))
        self._scaling = nn.Parameter(torch.cat([self._scaling, gaussians._scaling.cuda()]))
        self._rotation = nn.Parameter(torch.cat([self._rotation, gaussians._rotation.cuda()]))
        self._opacity = nn.Parameter(torch.cat([self._opacity, gaussians._opacity.cuda()]))
        self.max_radii2D = torch.cat([self.max_radii2D, gaussians.max_radii2D.cuda()])
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, gaussians.xyz_gradient_accum.cuda()])
        self.xyz_gradient_accum_abs = torch.cat([self.xyz_gradient_accum_abs, gaussians.xyz_gradient_accum_abs.cuda()])
        self.t_gradient_accum = torch.cat([self.t_gradient_accum, gaussians.t_gradient_accum.cuda()])
        self.specular_time_gradient_accum = torch.cat([self.specular_time_gradient_accum, gaussians.specular_time_gradient_accum.cuda()])
        self.denom = torch.cat([self.denom, gaussians.denom.cuda()])
        #self.spatial_lr_scale = torch.cat([self.spatial_lr_scale, gaussians.spatial_lr_scale]).cuda()
        self._t = nn.Parameter(torch.cat([self._t, gaussians._t.cuda()]))
        self._scaling_t = nn.Parameter(torch.cat([self._scaling_t, gaussians._scaling_t.cuda()]))
        self._velocity = nn.Parameter(torch.cat([self._velocity, gaussians._velocity.cuda()]))
        self._velocity2 = nn.Parameter(torch.cat([self._velocity2, gaussians._velocity2.cuda()]))
        self._velocity3 = nn.Parameter(torch.cat([self._velocity3, gaussians._velocity3.cuda()]))
        self._rot_velocity = nn.Parameter(torch.cat([self._rot_velocity, gaussians._rot_velocity.cuda()]))
        self._specular = nn.Parameter(torch.cat([self._specular, gaussians._specular.cuda()]))
        self._albedo = nn.Parameter(torch.cat([self._albedo, gaussians._albedo.cuda()]))
        self._specular2 = nn.Parameter(torch.cat([self._specular2, gaussians._specular2.cuda()]))
        self._delta_normal = nn.Parameter(torch.cat([self._delta_normal, gaussians._delta_normal.cuda()]))
        self._roughness = nn.Parameter(torch.cat([self._roughness, gaussians._roughness.cuda()]))
        # if gaussians.env_map is not None:
        #     self.env_map = torch.cat([self.env_map, gaussians.env_map.cuda()])

    def append_state_from_gaussian_cpu(self, gaussians : "GaussianModel", state_dict):
        for k, v in gaussians.opt_states.items():
            #optimizer_state = gaussians.optimizer.state.get(group["params"][0], None)
            state_content = {}
            state_content["step"] = v["step"].cuda()
            state_content["exp_avg"] = v["exp_avg"].cuda()
            state_content["exp_avg_sq"] = v["exp_avg_sq"].cuda()
            attr = self.get_param_group_corresponding_attr(k)
            self_state = self.optimizer.state.get(attr, None)
            if self_state is not None and len(self_state) != 0:
                self_state["step"] = state_content["step"]
                self_state["exp_avg"] = torch.cat([self_state["exp_avg"], state_content["exp_avg"]])
                self_state["exp_avg_sq"] = torch.cat([self_state["exp_avg_sq"], state_content["exp_avg_sq"]])
            else:
                if state_dict is not None:
                    previous_state = state_dict[k]
                    previous_state["step"] = state_content["step"]
                    previous_state["exp_avg"] = state_content["exp_avg"]
                    previous_state["exp_avg_sq"] = state_content["exp_avg_sq"]
                    self.optimizer.state[attr] = previous_state

    def clone_state_from_gaussians_cpu(self, gaussians_segments, state_dict):
        step_dict = {}
        exp_avg_dict = {}
        exp_avg_sq_dict = {}
        for segment in gaussians_segments:
            for k,v in segment.opt_states.items():
                if k not in step_dict:
                    step_dict[k] = []
                if k not in exp_avg_dict:
                    exp_avg_dict[k] = []
                if k not in exp_avg_sq_dict:
                    exp_avg_sq_dict[k] = []
                step_dict[k].append(v["step"].cuda())
                exp_avg_dict[k].append(v["exp_avg"].cuda())
                exp_avg_sq_dict[k].append(v["exp_avg_sq"].cuda())
        for k,v in gaussians_segments[0].opt_states.items():
            attr = self.get_param_group_corresponding_attr(k)
            if k in state_dict:
                previous_state = state_dict[k]
                #previous_state["step"] = step_dict[k][0]
                previous_state["exp_avg"] = torch.cat(exp_avg_dict[k])
                previous_state["exp_avg_sq"] = torch.cat(exp_avg_sq_dict[k])
                self.optimizer.state[attr] = previous_state

    def append_state_from_gaussians_cpu(self, gaussians_segments, state_dict):
        step_dict = {}
        exp_avg_dict = {}
        exp_avg_sq_dict = {}
        for segment in gaussians_segments:
            for k,v in segment.opt_states.items():
                if k not in step_dict:
                    step_dict[k] = []
                if k not in exp_avg_dict:
                    exp_avg_dict[k] = []
                if k not in exp_avg_sq_dict:
                    exp_avg_sq_dict[k] = []
                step_dict[k].append(v["step"].cuda())
                exp_avg_dict[k].append(v["exp_avg"].cuda())
                exp_avg_sq_dict[k].append(v["exp_avg_sq"].cuda())
        for k,v in gaussians_segments[0].opt_states.items():
            attr = self.get_param_group_corresponding_attr(k)
            if k in state_dict:
                previous_state = state_dict[k]
                #previous_state["step"] = step_dict[k][0]
                previous_state["exp_avg"] = torch.cat([previous_state["exp_avg"], torch.cat(exp_avg_dict[k])])
                previous_state["exp_avg_sq"] = torch.cat([previous_state["exp_avg_sq"], torch.cat(exp_avg_sq_dict[k])])
                self.optimizer.state[attr] = previous_state
                
    def clone_opt_states_from_gaussians_cpu(self, gaussians_segments):
        for segment in gaussians_segments:
            for k,v in segment.opt_states.items():
                if k not in self.opt_states:
                    self.opt_states[k] = {}
                    self.opt_states[k]["exp_avg"] = v["exp_avg"]
                    self.opt_states[k]["exp_avg_sq"] = v["exp_avg_sq"]
                    self.opt_states[k]["step"] = v["step"]
                else:
                    self.opt_states[k]["exp_avg"] = torch.cat([self.opt_states[k]["exp_avg"], v["exp_avg"]])
                    self.opt_states[k]["exp_avg_sq"] = torch.cat([self.opt_states[k]["exp_avg_sq"], v["exp_avg_sq"]])
        
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) + 0.001
    
    @property
    def get_scaling_t(self):
        return self.scaling_activation(self._scaling_t)
    
    @property
    def get_sigma_t(self):
        return self.scaling_activation(self._scaling_t) ** 2

    @property
    def get_sigma_t_fixed(self):
        # sigma_t = self.scaling_activation(self._scaling_t) ** 2
        # if getattr(self, "temporal_opacity_mode", "normalized_sigmoid") == "flat_window":
        #     sigma_t = sigma_t + self.get_temporal_flat_radius()
        # return torch.clip(sigma_t, min=1.0)
        return self.get_xyz.new_ones((self.get_xyz.shape[0], 1))
    
    @property
    def get_scaling_xyzt(self):
        return self.scaling_activation(torch.cat([self._scaling, self._scaling_t], dim = 1))
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_velocity(self):
        return self._velocity
    
    @property
    def get_velocity2(self):
        # return self._velocity2
        #return torch.exp(torch.clamp(self._velocity2, max=3.0))
        return torch.clamp(self._velocity2, max=12.0)
    
    @property
    def get_velocity3(self):
        return self._velocity3
    
    @property
    def get_rot_velocity(self):
        return self._rot_velocity
    
    @property
    def get_specular(self):
        return self.specular_activation(self._specular)
        #return torch.exp(torch.clamp(self._specular, max=3.0))
        # bias = torch.tensor(5.0, dtype=torch.float32).to("cuda")
        # return torch.exp(torch.clamp(self._specular, max=5.0))

    @property
    def get_albedo(self):
        #return self._albedo
        bias = torch.tensor(5.0, dtype=torch.float32).to("cuda")
        return torch.exp(torch.clamp(self._albedo, max=5.0)-torch.log(bias))

    @property
    def get_specular2(self):
        return self.specular2_activation(self._specular2)
    
    @property
    def get_delta_normal(self):
        return self._delta_normal

    @property
    def get_roughness(self):
        return self.roughness_activation(self._roughness)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_t(self):
        return self._t
    
    @property
    def get_xyzt(self):
        return torch.cat([self._xyz, self._t], dim = 1)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_specular2_temporal_variation(self):
        if not (self.gaussian_dim == 4 and self.rot_4d):
            return torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        num_points = self.get_xyz.shape[0]
        variation = torch.zeros((num_points, 1), device=self.device)

        static_grad_metric = None
        coeff_grad_metric = None

        # Static BRDF features are stored in _specular, while temporal coefficients are in _specular2.
        if self._specular.numel() > 0 and self._specular.grad is not None:
            static_grad_metric = self._specular.grad.abs().mean(dim=1, keepdim=True)
            if static_grad_metric.shape[0] == num_points:
                variation += static_grad_metric.to(variation.device)
            else:
                static_grad_metric = None

        if self._specular2.numel() > 0 and self._specular2.grad is not None:
            feature_coeff_grad = self._specular2.grad[:, self.gsdim:]
            if feature_coeff_grad.shape[1] == 0:
                feature_coeff_grad = self._specular2.grad
            if feature_coeff_grad.shape[1] > 0:
                coeff_grad_metric = feature_coeff_grad.abs().mean(dim=1, keepdim=True)
                if coeff_grad_metric.shape[0] == num_points:
                    variation += coeff_grad_metric.to(variation.device)
                else:
                    coeff_grad_metric = None

        if static_grad_metric is not None and coeff_grad_metric is not None:
            variation = 0.5 * variation

        return variation
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_max_sh_channels(self):
        if self.gaussian_dim == 3 or self.force_sh_3d:
            return (self.max_sh_degree+1)**2
        elif self.gaussian_dim == 4 and self.max_sh_degree_t == 0:
            return sh_channels_4d[self.max_sh_degree]
        elif self.gaussian_dim == 4 and self.max_sh_degree_t > 0:
            return (self.max_sh_degree+1)**2 * (self.max_sh_degree_t + 1)
    
    def get_diffuse(self, dir):
        return torch.clamp_min(eval_sh(0, self._features_dc.transpose(1, 2), None) + 0.5, 0)
        # return eval_sh(0, self._features_dc.transpose(1, 2), None) + 0.5


        #return eval_sh(2, self.get_features.transpose(1, 2), dir)
    
    def get_marginal_t(self, timestamp, scaling_modifier = 1): # Standard
        sigma = self.get_sigma_t * scaling_modifier ** 2
        return torch.exp(-0.5*(self.get_t-timestamp)**2/sigma) # / torch.sqrt(2*torch.pi*sigma)
    

    def get_temporal_opacity_factor(self, timestamp, scaling_modifier = 1):
        marginal_t = self.get_marginal_t(timestamp, scaling_modifier)
        mode = getattr(self, "temporal_opacity_mode", "normalized_sigmoid")
        if mode == "gaussian" or not (self.gaussian_dim == 4 and self.rot_4d):
            return marginal_t

        if mode == "flat_window":
            flat_radius, edge_sigma = self.get_temporal_flat_window(scaling_modifier)
            dt = torch.abs(self.get_t - timestamp)
            edge_dt = torch.clamp_min(dt - flat_radius, 0.0)
            return torch.exp(-0.5 * (edge_dt / torch.clamp_min(edge_sigma, 1.0e-6)) ** 2)

        k = self.get_temporal_opacity_slope()
        if mode == "pure_sigmoid":
            return torch.sigmoid((marginal_t - 0.5) * k)
        if mode == "normalized_sigmoid":
            min_opa = torch.sigmoid(-0.5 * k)
            scaler = torch.sigmoid(0.5 * k) - min_opa
            return (torch.sigmoid((marginal_t - 0.5) * k) - min_opa) / torch.clamp_min(scaler, 1.0e-6)

        raise ValueError(f"Unknown temporal_opacity_mode: {mode}")

    def get_temporal_opacity_slope(self):
        return torch.clamp(torch.abs(self.get_velocity2[..., 0:1]), min=1.0e-4)

    def get_temporal_edge_sigma(self, scaling_modifier = 1):
        return torch.clamp_min((self.get_scaling_t ** 2) * scaling_modifier, 1.0e-6)

    def get_temporal_flat_radius(self, scaling_modifier = 1):
        scale_t = torch.clamp_min(self.get_scaling_t * scaling_modifier, 1.0e-6)
        if not (self.gaussian_dim == 4 and self.rot_4d) or self._velocity2.numel() == 0:
            return torch.zeros_like(scale_t)
        if self._velocity2.shape[-1] < 1:
            return torch.zeros_like(scale_t)
        # flat_radius = torch.nn.functional.softplus(torch.clamp(self._velocity2[..., 0:1], min=-20.0, max=20.0))
        flat_radius = self._velocity2[..., 0:1]
        return torch.clamp_min(flat_radius * scaling_modifier, 0.0)

    def temporal_flat_radius_inverse_activation(self, flat_radius):
        # flat_radius = torch.clamp_min(flat_radius, 1.0e-8)
        # return torch.where(flat_radius > 20.0, flat_radius, torch.log(torch.expm1(flat_radius)))
        return flat_radius

    # def get_temporal_flat_radius_mult(self):
    #     scale_t = torch.clamp_min(self.get_scaling_t, 1.0e-6)
    #     return self.get_temporal_flat_radius() / scale_t

    def get_temporal_flat_window(self, scaling_modifier = 1):
        flat_radius = self.get_temporal_flat_radius(scaling_modifier)
        edge_sigma = self.get_temporal_edge_sigma(scaling_modifier)
        return flat_radius, edge_sigma

    def get_temporal_flat_ratio(self, opacity_level=0.05):
        flat_radius = self.get_temporal_flat_radius()
        effect_range = self.get_temporal_range_for_opacity(opacity_level)
        return flat_radius / torch.clamp_min(effect_range, 1.0e-6)

    def get_temporal_marginal_for_opacity(self, opacity_level):
        k = self.get_temporal_opacity_slope()
        level = torch.as_tensor(opacity_level, dtype=k.dtype, device=k.device)
        mode = getattr(self, "temporal_opacity_mode", "normalized_sigmoid")

        if mode == "gaussian" or not (self.gaussian_dim == 4 and self.rot_4d):
            marginal_t = level.expand_as(k)
        elif mode == "flat_window":
            flat_radius, edge_sigma = self.get_temporal_flat_window()
            target = torch.clamp(level.expand_as(k), min=1.0e-6, max=1.0)
            temporal_range = flat_radius + edge_sigma * torch.sqrt(-2.0 * torch.log(target))
            marginal_t = torch.exp(-0.5 * ((temporal_range - flat_radius) / torch.clamp_min(edge_sigma, 1.0e-6)) ** 2)
        elif mode == "pure_sigmoid":
            target = torch.clamp(level.expand_as(k), min=1.0e-6, max=1.0 - 1.0e-6)
            marginal_t = inverse_sigmoid(target) / k + 0.5
        elif mode == "normalized_sigmoid":
            min_opa = torch.sigmoid(-0.5 * k)
            scaler = torch.sigmoid(0.5 * k) - min_opa
            target = torch.clamp(level * scaler + min_opa, min=1.0e-6, max=1.0 - 1.0e-6)
            marginal_t = inverse_sigmoid(target) / k + 0.5
        else:
            raise ValueError(f"Unknown temporal_opacity_mode: {mode}")

        return torch.clamp(marginal_t, min=1.0e-6, max=1.0 - 1.0e-6)

    def get_temporal_range_for_opacity(self, opacity_level):
        mode = getattr(self, "temporal_opacity_mode", "normalized_sigmoid")
        if mode == "flat_window" and self.gaussian_dim == 4 and self.rot_4d:
            flat_radius, edge_sigma = self.get_temporal_flat_window()
            target = torch.as_tensor(opacity_level, dtype=flat_radius.dtype, device=flat_radius.device)
            target = torch.clamp(target.expand_as(flat_radius), min=1.0e-6, max=1.0)
            return flat_radius + edge_sigma * torch.sqrt(-2.0 * torch.log(target))

        marginal_t = self.get_temporal_marginal_for_opacity(opacity_level)
        range_sq = torch.clamp_min(-2.0 * torch.log(marginal_t), 0.0)
        return torch.clamp_min(self.get_scaling_t, 1.0e-6) * torch.sqrt(range_sq)

    def get_temporal_split_params(self, selected_pts_mask, N=2):
        N = int(N)
        if N < 2:
            raise ValueError(f"Temporal split requires N >= 2, got {N}")

        parent_t = self.get_t[selected_pts_mask]
        parent_scaling_t = torch.clamp_min(self.get_scaling_t[selected_pts_mask], 1.0e-6)
        mode = getattr(self, "temporal_opacity_mode", "normalized_sigmoid")

        if mode == "flat_window":
            # Keep edge sigma unchanged after time split.  Since edge sigma is
            # derived from scaling_t, each child keeps the parent's scaling_t.
            child_scaling_t = parent_scaling_t
            child_center_offset = self.get_temporal_flat_radius()[selected_pts_mask] / float(N)
        else:
            child_scaling_t = parent_scaling_t / float(N)
            parent_peak = self.get_temporal_opacity_factor(self.get_t).detach()
            center_level = torch.clamp(parent_peak / float(N), min=1.0e-6, max=1.0 - 1.0e-6)
            child_center_offset = self.get_temporal_range_for_opacity(center_level)[selected_pts_mask] / float(N)

        child_scaling_t_log = self.scaling_inverse_activation(child_scaling_t)
        split_index = torch.arange(N, dtype=parent_t.dtype, device=parent_t.device) - (N - 1) / 2.0
        time_offsets = (2.0 * split_index[:, None, None] * child_center_offset[None]).reshape(-1, 1)

        new_t = parent_t.repeat(N, 1) + time_offsets
        new_scaling_t = child_scaling_t_log.repeat(N, 1)
        return new_t, new_scaling_t, time_offsets, child_scaling_t

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def reset_opacity_large(self):
        opacities_new = self.get_opacity.detach()
        mask = opacities_new > 0.95
        if mask.any():
            new_val = self.inverse_opacity_activation(torch.tensor(0.8, device=self.device))
            self._opacity.data[mask] = new_val

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
        elif self.max_sh_degree_t and self.active_sh_degree_t < self.max_sh_degree_t:
            self.active_sh_degree_t += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        albedo = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        if self.gaussian_dim == 4:
            if pcd.time is None:
                fused_times = (torch.rand(fused_point_cloud.shape[0], 1, device="cuda") * 1.2 - 0.1) * (self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]
            else:
                fused_times = torch.from_numpy(pcd.time).cuda().float()
            
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        if self.gaussian_dim == 4:
            # dist_t = torch.clamp_min(distCUDA2(fused_times.repeat(1,3)), 1e-10)[...,None]
            dist_t = torch.zeros_like(fused_times, device="cuda") + (self.time_duration[1] - self.time_duration[0])
            scales_t = torch.log(0.4 * dist_t)
            if self.rot_4d:
                velocity = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
                velocity2 = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
                velocity3 = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
                rot_velocity = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
                specular = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
                specular2 = torch.zeros((fused_point_cloud.shape[0], 44), device="cuda")
                #specular2 = torch.zeros((fused_point_cloud.shape[0], 20), device="cuda")
                #specular2 = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
                delta_normal = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
                roughness = self.default_roughness * torch.ones((fused_point_cloud.shape[0], 1), device="cuda")

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        if self.gaussian_dim == 4:
            self._t = nn.Parameter(fused_times.requires_grad_(True))
            self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
            if self.rot_4d:
                self._velocity = nn.Parameter(velocity.requires_grad_(True))
                self._velocity2 = nn.Parameter(velocity2.requires_grad_(True))
                self._velocity3 = nn.Parameter(velocity3.requires_grad_(True))
                self._rot_velocity = nn.Parameter(rot_velocity.requires_grad_(True))
                self._specular = nn.Parameter(specular.requires_grad_(True))
                self._albedo = nn.Parameter(albedo.requires_grad_(True))
                self._specular2 = nn.Parameter(specular2.requires_grad_(True))
                self._delta_normal = nn.Parameter(delta_normal.requires_grad_(True))
                self._roughness = nn.Parameter(roughness.requires_grad_(True))

        # self.spatial_lr_scale = spatial_lr_scale
        # fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float()
        # fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float())
        # features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float()
        # features[:, :3, 0 ] = fused_color
        # features[:, 3:, 1:] = 0.0
        # if self.gaussian_dim == 4:
        #     if pcd.time is None:
        #         fused_times = (torch.rand(fused_point_cloud.shape[0], 1, device=self.device) * 1.2 - 0.1) * (self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]
        #     else:
        #         fused_times = torch.from_numpy(pcd.time).float()
            
        # print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float()), 0.0000001)
        # scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        # rots[:, 0] = 1
        # if self.gaussian_dim == 4:
        #     # dist_t = torch.clamp_min(distCUDA2(fused_times.repeat(1,3)), 1e-10)[...,None]
        #     dist_t = torch.zeros_like(fused_times, device=self.device) + (self.time_duration[1] - self.time_duration[0]) / 5
        #     scales_t = torch.log(torch.sqrt(dist_t))
        #     if self.rot_4d:
        #         velocity = torch.zeros((fused_point_cloud.shape[0], 3), device=self.device)

        # opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device))

        # self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # self._scaling = nn.Parameter(scales.requires_grad_(True))
        # self._rotation = nn.Parameter(rots.requires_grad_(True))
        # self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)
        
        # if self.gaussian_dim == 4:
        #     self._t = nn.Parameter(fused_times.requires_grad_(True))
        #     self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
        #     if self.rot_4d:
        #         self._velocity = nn.Parameter(velocity.requires_grad_(True))

    def create_from_pcd_cpu(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float())
        albedo = torch.tensor(np.asarray(pcd.colors)).float()
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        if self.gaussian_dim == 4:
            if pcd.time is None:
                fused_times = (torch.rand(fused_point_cloud.shape[0], 1, device=self.device) * 1.2 - 0.1) * (self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]
            else:
                fused_times = torch.from_numpy(pcd.time).float()
            
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001).to(self.device)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
        rots[:, 0] = 1
        if self.gaussian_dim == 4:
            # dist_t = torch.clamp_min(distCUDA2(fused_times.repeat(1,3)), 1e-10)[...,None]
            dist_t = torch.zeros_like(fused_times, device=self.device) + (self.time_duration[1] - self.time_duration[0]) / 5
            scales_t = torch.log(torch.sqrt(dist_t))
            if self.rot_4d:
                velocity = torch.zeros((fused_point_cloud.shape[0], 3), device=self.device)
                velocity2 = torch.zeros((fused_point_cloud.shape[0], 3), device=self.device)
                velocity3 = torch.zeros((fused_point_cloud.shape[0], 3), device=self.device)
                rot_velocity = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
                specular = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
                specular2 = torch.zeros((fused_point_cloud.shape[0], 44), device=self.device)
                #specular2 = torch.zeros((fused_point_cloud.shape[0], 20), device=self.device)
                #specular2 = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
                delta_normal = torch.zeros((fused_point_cloud.shape[0], 3), device=self.device)
                roughness = self.default_roughness * torch.ones((fused_point_cloud.shape[0], 1), device=self.device)

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)
        
        if self.gaussian_dim == 4:
            self._t = nn.Parameter(fused_times.requires_grad_(True))
            self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
            if self.rot_4d:
                self._velocity = nn.Parameter(velocity.requires_grad_(True))
                self._velocity2 = nn.Parameter(velocity2.requires_grad_(True))
                self._velocity3 = nn.Parameter(velocity3.requires_grad_(True))
                self._rot_velocity = nn.Parameter(rot_velocity.requires_grad_(True))
                self._specular = nn.Parameter(specular.requires_grad_(True))
                self._albedo = nn.Parameter(albedo.requires_grad_(True))
                self._specular2 = nn.Parameter(specular2.requires_grad_(True))
                self._delta_normal = nn.Parameter(delta_normal.requires_grad_(True))
                self._roughness = nn.Parameter(roughness.requires_grad_(True))

    def create_from_multi_pcd(self, path, tgh, spatial_lr_scale : float, time_duration=None, max_points=None, frame_filter=None):
        self.spatial_lr_scale = spatial_lr_scale
        self._xyz = None
        pcd_parent_path = os.path.join(path, 'pcds_j10')
        pcd_list = os.listdir(os.path.join(path, 'pcds_j10'))
        #pcd_list = ["points3d.ply" for i in range(200)]
        pcd_list = sorted(pcd_list)
        fps = 30.
        temporal_init_range_level = 0.05
        temporal_init_edge_range = 0.5 / fps
        temporal_init_edge_sigma = temporal_init_edge_range / math.sqrt(-2.0 * math.log(temporal_init_range_level))
        temporal_init_scaling_t = math.sqrt(temporal_init_edge_sigma)
        temporal_init_scaling_t_log = math.log(max(temporal_init_scaling_t, 1.0e-8))
        # The renderer stores flat_radius as a half-width.  A one-frame flat
        # region therefore starts with a half-width of 0.5 / fps.
        temporal_init_flat_radius = 0.5 / fps
        # temporal_init_flat_raw = math.log(math.expm1(max(temporal_init_flat_radius, 1.0e-8)))
        temporal_init_flat_raw = temporal_init_flat_radius
        temporal_static_init_flat_radius = 30.0 / fps
        # temporal_static_init_flat_raw = math.log(math.expm1(max(temporal_static_init_flat_radius, 1.0e-8)))
        temporal_static_init_flat_raw = temporal_static_init_flat_radius
        #temporal_init_sigma_t_fixed = max(temporal_init_edge_sigma + temporal_init_flat_radius, 1.0)
        temporal_init_sigma_t_fixed = 1.0
        print(
            "Temporal init:",
            "flat_width", 1.0 / fps,
            "flat_radius", temporal_init_flat_radius,
            "edge_range_0.05", temporal_init_edge_range,
            "edge_sigma", temporal_init_edge_sigma,
            "scaling_t", temporal_init_scaling_t,
            "sigma_t_fixed", temporal_init_sigma_t_fixed,
            "flat_raw", temporal_init_flat_raw,
            "static_flat_radius", temporal_static_init_flat_radius,
            "static_flat_raw", temporal_static_init_flat_raw,
        )
        frame_filter_set = _parse_frame_filter(frame_filter)
        points_before = None
        selected_frame_count = 0
        for frame_idx, _ in enumerate(pcd_list[:]):
            if frame_idx < 81 and frame_idx >= 19:
                if frame_filter_set is not None and frame_idx not in frame_filter_set:
                    continue
                timestamp = frame_idx / fps
                if time_duration is not None and (timestamp < time_duration[0] or timestamp > time_duration[1]):
                    continue
                selected_frame_count += 1
        frame_point_limit = 50000
        if max_points is not None and selected_frame_count > 0:
            frame_point_limit = max(1, min(frame_point_limit, int(math.ceil(max_points / selected_frame_count))))
        with torch.no_grad():
        # pcd_list = ['points.ply' for _ in range(640)]
            for ii, pcd_path in enumerate(pcd_list[:]):
                #if ii < 1:
                
                if ii < 81 and ii >= 19:
                    if frame_filter_set is not None and ii not in frame_filter_set:
                        continue
                #if ii < 60 and ii >= 0:
                #if ii == 71:
                #if ii < 10:
            
                    # if ii % 30 != 0:
                    #     continue
                    timestamp = (ii) / fps
                    # timestamp = ii / fps
                    if time_duration is not None:
                        if timestamp < time_duration[0]:
                            continue
                        if timestamp > time_duration[1]:
                            break
                    #break
                    ply_path = os.path.join(pcd_parent_path, pcd_path)
                    pcd = fetchPly(ply_path)
                    if pcd.points.shape[0] > frame_point_limit:
                        mask = np.random.randint(0, pcd.points.shape[0], frame_point_limit)
                        xyz = pcd.points[mask]
                        rgb = pcd.colors[mask]
                        normals = pcd.normals[mask]

                        pcd = BasicPointCloud(points=xyz, colors=rgb, normals=normals, time=timestamp)
                    
                    fused_point_cloud = torch.tensor(np.asarray(pcd.points), device=self.device, dtype=torch.float)
                    fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors), device=self.device, dtype=torch.float))
                    #albedo = torch.tensor(np.asarray(pcd.colors), device=self.device, dtype=torch.float)
                    albedo = torch.zeros((fused_color.shape[0], 3), device=self.device, dtype=torch.float)
                    features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels), device=self.device, dtype=torch.float)
                    features[:, :3, 0 ] = fused_color
                    features[:, 3:, 1:] = 0.0
                    if self.gaussian_dim == 4:
                        # seg = 2.5/4
                        seg = 1 / fps * 3
                        # fused_times = torch.zeros_like(fused_point_cloud[..., :1]) + ((timestamp + seg/4 - time_duration[0]) // seg) * seg + seg/2 + time_duration[0] - seg/4
                        fused_times = (torch.zeros_like(fused_point_cloud[..., :1]) + timestamp - time_duration[0]) / 1
                    print("Number of points at initialisation : ", fused_point_cloud.shape[0], timestamp - time_duration[0])

                    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001).to(self.device)
                    if points_before is None:
                        dist2b = fused_point_cloud
                    else:
                        dist2b = distCUDA2b(torch.from_numpy(np.asarray(pcd.points)).float().cuda(),points_before).to(self.device)
                    scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
                    rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
                    rots[:, 0] = 1
                    if self.gaussian_dim == 4:
                        # dist_t = torch.clamp_min(distCUDA2(fused_times.repeat(1,3)), 1e-10)[...,None]
                        # dist_t = torch.zeros_like(fused_times, device=self.device) + (self.time_duration[1] - self.time_duration[0]) / 1000
                        dist_t = (torch.zeros_like(fused_times, device=self.device) + seg / 2) / 1
                        # scaling_t is initialized so the edge opacity remains above
                        # 0.05 for about half a frame outside the flat region.
                        scales_t = torch.full_like(fused_times, temporal_init_scaling_t_log, device=self.device)
                        if self.rot_4d:
                            velocity = (fused_point_cloud - dist2b) * fps * temporal_init_sigma_t_fixed
                            velocity2 = torch.zeros((fused_point_cloud.shape[0], 3), device=self.device)
                            velocity2[..., 0:1] = temporal_init_flat_raw
                            velocity3 = torch.zeros((fused_point_cloud.shape[0], 3), device=self.device)
                            rot_velocity = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
                            specular = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
                            specular2 = torch.zeros((fused_point_cloud.shape[0], 44), device=self.device)
                            #specular2 = torch.zeros((fused_point_cloud.shape[0], 20), device=self.device)
                            #specular2 = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
                            delta_normal = torch.zeros((fused_point_cloud.shape[0], 3), device=self.device)
                            roughness = self.default_roughness * torch.ones((fused_point_cloud.shape[0], 1), device=self.device)

                    opacities = inverse_sigmoid(0.2 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device))

                    self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
                    self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
                    self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
                    self._scaling = nn.Parameter(scales.requires_grad_(True))
                    self._rotation = nn.Parameter(rots.requires_grad_(True))
                    self._opacity = nn.Parameter(opacities.requires_grad_(True))
                    
                    if self.gaussian_dim == 4:
                        self._t = nn.Parameter(fused_times.requires_grad_(True))
                        self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
                        if self.rot_4d:
                            self._velocity = nn.Parameter(velocity.requires_grad_(True))
                            self._velocity2 = nn.Parameter(velocity2.requires_grad_(True))
                            self._velocity3 = nn.Parameter(velocity3.requires_grad_(True))
                            self._rot_velocity = nn.Parameter(rot_velocity.requires_grad_(True))
                            self._specular = nn.Parameter(specular.requires_grad_(True))
                            self._albedo = nn.Parameter(albedo.requires_grad_(True))
                            self._specular2 = nn.Parameter(specular2.requires_grad_(True))
                            self._delta_normal = nn.Parameter(delta_normal.requires_grad_(True))
                            self._roughness = nn.Parameter(roughness.requires_grad_(True))

                    self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)
                    # self.xyz_gradient_accum = torch.zeros_like(self._xyz, device=self.device)
                    # self.t_gradient_accum = torch.zeros_like(self._t, device=self.device)
                    # self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
                    self.training_setup(tgh.opt)
                    tgh.create_from_gaussians(self)
                    # del self._xyz
                    # del self._features_dc
                    # del self._features_rest
                    # del self._scaling
                    # del self._rotation
                    # del self._opacity
                    # del self._t
                    # del self._scaling_t
                    # del self._velocity
                    # del fused_point_cloud
                    # del fused_color
                    # del fused_times
                    # del features
                    # del dist2
                    # del scales
                    # del rots
                    # del dist_t
                    # del scales_t
                    # del velocity
                    # del opacities
                    # del self.xyz_gradient_accum
                    # del self.xyz_gradient_accum_abs
                    # del self.denom
                    # del self.optimizer
                    # gc.collect()
                    #torch.cuda.empty_cache()
                    points_before = fused_point_cloud

        ply_path = os.path.join(path, 'points3d.ply')
        pcd = fetchPly(ply_path)
        final_point_limit = 500000 if max_points is None else max(1, int(max_points))
        if pcd.points.shape[0] > final_point_limit:
            mask = np.random.randint(0, pcd.points.shape[0], final_point_limit)
            xyz = pcd.points[mask]
            rgb = pcd.colors[mask]
            normals = pcd.normals[mask]

            pcd = BasicPointCloud(points=xyz, colors=rgb, normals=normals, time=None)
        
        fused_point_cloud = torch.tensor(np.asarray(pcd.points), device=self.device, dtype=torch.float)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors), device=self.device, dtype=torch.float))
        #albedo = torch.tensor(np.asarray(pcd.colors), device=self.device, dtype=torch.float)
        albedo = torch.zeros((fused_color.shape[0], 3), device=self.device, dtype=torch.float)
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels), device=self.device, dtype=torch.float)
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
            
        for ii in range(1):
            # timestamp = -10 / 2 / 2 + 5 * (2 * ii + 1)
            #timestamp = (self.time_duration[1] - self.time_duration[0]) / 2
            #timestamp = (2.3333333333333335 + 2.6333333333333333) / 2
            #timestamp = (2.3333333333333335 + 2.4) / 2
            #timestamp = 2.4
            #timestamp = (1.6666666666666667 + 2.6333333333333333) / 2
            #timestamp = (1.9666666666666666 + 0.0) / 2
            timestamp = (2.6333333333333333 + 0.6666666666666666) / 2
            #timestamp = (0.03333333333333333 + 0.03333333333333333) / 2
            # print(timestamp, 'sdfdfdfd')
            # if time_duration is not None:
            #     if timestamp < time_duration[0] - 10 / 2 / 2 or timestamp > time_duration[1] + 10 / 2 / 2:
            #         break
            if self.gaussian_dim == 4:
                fused_times = (torch.zeros_like(fused_point_cloud[..., :1]) + timestamp) / 1
                
            print("Number of points at initialisation : ", fused_point_cloud.shape[0])

            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001).to(self.device)
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
            rots[:, 0] = 1
            if self.gaussian_dim == 4:
                # dist_t = torch.clamp_min(distCUDA2(fused_times.repeat(1,3)), 1e-10)[...,None]
                # dist_t = torch.zeros_like(fused_times, device=self.device) + (self.time_duration[1] - self.time_duration[0]) / 100
                #dist_t = (torch.zeros_like(fused_times, device=self.device) + 20) / 1
                #dist_t = torch.zeros_like(fused_times, device=self.device) + (1.9666666666666666 - 0.0) / 2
                dist_t = torch.zeros_like(fused_times, device=self.device) + (2.6333333333333333 - 0.6666666666666666) / 2
                #dist_t = torch.zeros_like(fused_times, device=self.device) + (2.4 - 2.3333333333333335) / 2
                # dist_t = torch.zeros_like(fused_times, device=self.device)
                # Static points use the same sharp edge, but a much wider
                # flat half-width so they cover the full 20-80 frame window.
                scales_t = torch.full_like(fused_times, temporal_init_scaling_t_log, device=self.device)
                if self.rot_4d:
                    velocity = torch.zeros((fused_point_cloud.shape[0], 3), device=self.device)
                    velocity2 = torch.zeros((fused_point_cloud.shape[0], 3), device=self.device)
                    velocity2[..., 0:1] = temporal_static_init_flat_raw
                    velocity3 = torch.zeros((fused_point_cloud.shape[0], 3), device=self.device)
                    rot_velocity = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
                    specular = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
                    specular2 = torch.zeros((fused_point_cloud.shape[0], 44), device=self.device)
                    #specular2 = torch.zeros((fused_point_cloud.shape[0], 20), device=self.device)
                    #specular2 = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
                    delta_normal = torch.zeros((fused_point_cloud.shape[0], 3), device=self.device)
                    roughness = self.default_roughness * torch.ones((fused_point_cloud.shape[0], 1), device=self.device)
                    
            #print("test1")
            opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device))

            _xyz = fused_point_cloud
            _features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
            _features_rest = features[:,:,1:].transpose(1, 2).contiguous()
            _scaling = scales
            _rotation = rots
            _opacity = opacities
            self.max_radii2D = torch.zeros(_xyz.shape[0], device=self.device)
            #print("test2")
            if self.gaussian_dim == 4:
                _t = fused_times
                _scaling_t = scales_t
                if self.rot_4d:
                    _velocity = velocity
                    _velocity2 = velocity2
                    _velocity3 = velocity3
                    _rot_velocity = rot_velocity
                    _specular = specular
                    _albedo = albedo
                    _specular2 = specular2
                    _delta_normal = delta_normal
                    _roughness = roughness

            # tgh.create_from_gaussians(self)
        #print("test3")
        self._xyz = nn.Parameter(_xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(_features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(_features_rest.requires_grad_(True))
        # self._scaling += math.log(6)
        self._scaling = nn.Parameter(_scaling.requires_grad_(True))
        self._rotation = nn.Parameter(_rotation.requires_grad_(True))
        self._opacity = nn.Parameter(_opacity.requires_grad_(True))
        self.max_radii2D = torch.zeros(self.get_xyz.shape[0], device=self.device)
        
        if self.gaussian_dim == 4:
            self._t = nn.Parameter(_t.requires_grad_(True))
            self._scaling_t = nn.Parameter(_scaling_t.requires_grad_(True))
            if self.rot_4d:
                self._velocity = nn.Parameter(_velocity.requires_grad_(True))
                self._velocity2 = nn.Parameter(_velocity2.requires_grad_(True))
                self._velocity3 = nn.Parameter(_velocity3.requires_grad_(True))
                self._rot_velocity = nn.Parameter(_rot_velocity.requires_grad_(True))
                self._specular = nn.Parameter(_specular.requires_grad_(True))
                self._albedo = nn.Parameter(_albedo.requires_grad_(True))
                self._specular2 = nn.Parameter(_specular2.requires_grad_(True))
                self._delta_normal = nn.Parameter(_delta_normal.requires_grad_(True))
                self._roughness = nn.Parameter(_roughness.requires_grad_(True))
        # print(self.get_cov_t())
        # print(torch.sqrt(-math.log(0.05)/0.5*self.get_sigma_t[..., 0]).max())
        #print("test4")

    def save_ply_w_cnts2(self, path, cnts, roots):
        os.makedirs(os.path.dirname(path), exist_ok = True)

        xyz = self._xyz.detach().cpu().numpy().astype(np.float16)
        # normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy().astype(np.float16)
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy().astype(np.float16)
        opacities = self._opacity.detach().cpu().numpy().astype(np.float16)
        scale = self._scaling.detach().cpu().numpy().astype(np.float16)
        rotation = self._rotation.detach().cpu().numpy().astype(np.float16)
        t = self._t.detach().cpu().numpy().astype(np.float16)
        scale_t = self._scaling_t.detach().cpu().numpy().astype(np.float16)
        velocity = self._velocity.detach().cpu().numpy().astype(np.float16)
        omega = np.zeros_like(rotation).astype(np.float16)
        all_attr = {'num_segments_every_layer': np.array(roots, dtype=np.uint32), 'num_points_segments': np.array(cnts, dtype=np.uint32), 'max_segment_length': np.array(10.0, dtype=np.float32), 'mip': np.array(True, dtype=bool), 'kernel_size': np.array(0.1, dtype=np.float32), 'fps': np.array(30.0, dtype=np.float32), 'start_stamp': np.array(0.0, dtype=np.float32), 'end_stamp': np.array(1.0, dtype=np.float32),
            'xyz': xyz, 'f_dc': f_dc, 'f_rest': f_rest, 'opacities': opacities, 'scale': scale, 'rotation': rotation, 't': t, 'scale_t': scale_t, 'velocity': velocity, 'omega': omega}
        # np.save(path+'xxx.npy', all_attr)
        np.savez_compressed(path+'converted_gaussians.npz', **all_attr)

    def create_from_pth(self, path, spatial_lr_scale):
        assert self.gaussian_dim == 4 and self.rot_4d
        self.spatial_lr_scale = spatial_lr_scale
        init_4d_gaussian = torch.load(path)
        fused_point_cloud = init_4d_gaussian['xyz'].cuda()
        features_dc = init_4d_gaussian['features_dc'].cuda()
        features_rest = init_4d_gaussian['features_rest'].cuda()
        fused_times = init_4d_gaussian['t'].cuda()
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        scales = init_4d_gaussian['scaling'].cuda()
        rots = init_4d_gaussian['rotation'].cuda()
        scales_t = init_4d_gaussian['scaling_t'].cuda()
        velocity = init_4d_gaussian['velocity'].cuda()
        velocity2 = init_4d_gaussian['velocity2'].cuda()
        velocity3 = init_4d_gaussian['velocity3'].cuda()
        rot_velocity = init_4d_gaussian['rot_velocity'].cuda()
        specular = init_4d_gaussian['specular'].cuda()
        albedo = init_4d_gaussian['albedo'].cuda()
        specular2 = init_4d_gaussian['specular2'].cuda()
        delta_normal = init_4d_gaussian['delta_normal'].cuda()
        roughness = init_4d_gaussian['roughness'].cuda()

        opacities = init_4d_gaussian['opacity'].cuda()
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.transpose(1, 2).requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.transpose(1, 2).requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        self._t = nn.Parameter(fused_times.requires_grad_(True))
        self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
        self._velocity = nn.Parameter(velocity.requires_grad_(True))
        self._velocity2 = nn.Parameter(velocity2.requires_grad_(True))
        self._velocity3 = nn.Parameter(velocity3.requires_grad_(True))
        self._rot_velocity = nn.Parameter(rot_velocity.requires_grad_(True))
        self._specular = nn.Parameter(specular.requires_grad_(True))
        self._albedo = nn.Parameter(albedo.requires_grad_(True))
        self._specular2 = nn.Parameter(specular2.requires_grad_(True))
        self._delta_normal = nn.Parameter(delta_normal.requires_grad_(True))
        self._roughness = nn.Parameter(roughness.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.specular_time_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.gaussian_dim == 4: # TODO: tune time_lr_scale
            if training_args.position_t_lr_init < 0:
                training_args.position_t_lr_init = training_args.position_lr_init
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            velocity2_lr_init = getattr(training_args, "velocity2_lr_init", -1.0)
            velocity2_lr_final = getattr(training_args, "velocity2_lr_final", -1.0)
            velocity2_lr_delay_mult = getattr(training_args, "velocity2_lr_delay_mult", -1.0)
            velocity2_lr_max_steps = getattr(training_args, "velocity2_lr_max_steps", -1)
            if velocity2_lr_init < 0:
                velocity2_lr_init = training_args.scaling_lr
            if velocity2_lr_final < 0:
                velocity2_lr_final = velocity2_lr_init
            if velocity2_lr_delay_mult < 0:
                velocity2_lr_delay_mult = 1.0
            if velocity2_lr_max_steps < 0:
                velocity2_lr_max_steps = training_args.position_lr_max_steps
            self._velocity2_lr_init = velocity2_lr_init
            self._velocity2_lr_final = velocity2_lr_final
            self._velocity2_lr_delay_mult = velocity2_lr_delay_mult
            self._velocity2_lr_max_steps = velocity2_lr_max_steps
            l.append({'params': [self._t], 'lr': training_args.position_t_lr_init * self.spatial_lr_scale / 3, "name": "t"})
            l.append({'params': [self._scaling_t], 'lr': training_args.scaling_lr, "name": "scaling_t"})
            if self.rot_4d:
                l.append({'params': [self._velocity], 'lr': training_args.rotation_lr / 30, "name": "velocity"})
                l.append({'params': [self._velocity2], 'lr': velocity2_lr_init, "name": "velocity2"})
                l.append({'params': [self._velocity3], 'lr': training_args.rotation_lr / 50, "name": "velocity3"})
                l.append({'params': [self._rot_velocity], 'lr': training_args.rotation_lr, "name": "rot_velocity"})
                l.append({'params': [self._specular], 'lr': training_args.feature_lr * 5, "name": "specular"})
                l.append({'params': [self._albedo], 'lr': training_args.albedo_lr, "name": "albedo"})
                l.append({'params': [self._specular2], 'lr': training_args.feature_lr / 2, "name": "specular2"})
                l.append({'params': [self._delta_normal], 'lr': training_args.delta_normal_lr, "name": "delta_normal"})
                l.append({'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"})
                l.append({'params': list(self.brdf_mlp.parameters()), 'lr': training_args.brdf_mlp_lr_init, "name": "brdf_mlp"})
                l.append({'params': list(self.light_mlp.parameters()), 'lr': training_args.mlp_lr_init * 2, "name": "light_mlp"})
                l.append({'params': list(self.light_mlp_2.parameters()), 'lr': training_args.mlp_lr_init, "name": "light_mlp2"})
                l.append({'params': list(self.dir_encoding.parameters()), 'lr': training_args.encoding_lr_init, "name": "dir_encoding"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # self.optimizer = AdamWithMaskedUpdates(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.velocity_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr / 30,
                                                    lr_final=training_args.rotation_lr / 60,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.velocity2_scheduler_args = get_expon_lr_func(lr_init=self._velocity2_lr_init,
                                                    lr_final=self._velocity2_lr_final,
                                                    lr_delay_mult=self._velocity2_lr_delay_mult,
                                                    max_steps=self._velocity2_lr_max_steps)
        self.brdf_mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.brdf_mlp_lr_init,
                                        lr_final=training_args.brdf_mlp_lr_final,
                                        lr_delay_mult=training_args.brdf_mlp_lr_delay_mult,
                                        max_steps=training_args.brdf_mlp_lr_max_steps)
        self.light_mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_lr_init * 2,
                                        lr_final=training_args.mlp_lr_init,
                                        lr_delay_mult=training_args.mlp_lr_delay_mult,
                                        max_steps=training_args.mlp_lr_max_steps)
        self.light_mlp2_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_lr_init,
                                        lr_final=training_args.mlp_lr_init / 2,
                                        lr_delay_mult=training_args.mlp_lr_delay_mult,
                                        max_steps=training_args.mlp_lr_max_steps)
        self.encoding_scheduler_args = get_expon_lr_func(lr_init=training_args.encoding_lr_init,
                                        lr_final=training_args.encoding_lr_final,
                                        lr_delay_mult=training_args.encoding_lr_delay_mult,
                                        max_steps=training_args.encoding_lr_max_steps)
        self.t_scheduler_args = get_expon_lr_func(lr_init=training_args.position_t_lr_init * self.spatial_lr_scale / 3,
                                                    lr_final=training_args.position_t_lr_init * self.spatial_lr_scale / 12,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.albedo_scheduler_args = get_expon_lr_func(lr_init=training_args.albedo_lr,
                                        lr_final=training_args.albedo_lr / 2,
                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                        max_steps=training_args.position_lr_max_steps)
        # self.scaling_t_scheduler_args = get_expon_lr_func(lr_init=training_args.scaling_lr,
        #                                             lr_final=training_args.scaling_lr / 2,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.position_lr_max_steps)

        self.specular_feature_scheduler_args = get_expon_lr_func(lr_init=training_args.feature_lr * 5,
                                                    lr_final=training_args.feature_lr,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.specular_feature_coeff_scheduler_args = get_expon_lr_func(lr_init=training_args.feature_lr / 2,
                                                    lr_final=training_args.feature_lr / 4,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        # self.opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.opacity_lr * 2,
        #                                             lr_final=training_args.opacity_lr,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.position_lr_max_steps)
        


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                #return lr
            if param_group["name"] == "velocity":
                lr = self.velocity_scheduler_args(iteration)
                param_group["lr"] = lr
                #return lr
            if param_group["name"] == "brdf_mlp":
                lr = self.brdf_mlp_scheduler_args(iteration)
                param_group["lr"] =lr
                #return lr
            if param_group["name"] == "t":
                lr = self.t_scheduler_args(iteration)
                param_group["lr"] = lr
                #return lr
            # if param_group["name"] == "scaling_t":
            #     lr = self.scaling_t_scheduler_args(iteration)
            #     param_group["lr"] = lr
            #     #return lr
            if param_group["name"] == "specular":
                lr = self.specular_feature_scheduler_args(iteration)
                param_group["lr"] = lr
                #return lr
            if param_group["name"] == "specular2":
                lr = self.specular_feature_coeff_scheduler_args(iteration)
                param_group["lr"] = lr
                #return lr
            if param_group["name"] == "light_mlp":
                lr = self.light_mlp_scheduler_args(iteration)
                param_group["lr"] = lr
                #return lr
            if param_group["name"] == "light_mlp2":
                lr = self.light_mlp2_scheduler_args(iteration)
                param_group["lr"] = lr
                #return lr
            if param_group["name"] == "dir_encoding":
                lr = self.encoding_scheduler_args(iteration)
                param_group["lr"] = lr
                #return lr
            if param_group["name"] == "albedo":
                lr = self.albedo_scheduler_args(iteration)
                param_group["lr"] = lr
                #return lr
            # if param_group["name"] == "opacity":
            #     lr = self.opacity_scheduler_args(iteration)
            #     param_group["lr"] = lr
            #     #return lr
            if param_group["name"] == "velocity2":
                lr = self.velocity2_scheduler_args(iteration)
                param_group["lr"] = lr
                #return lr
            # if param_group["name"] == "velocity3":
            #     lr = self.velocity2_scheduler_args(iteration)
            #     param_group["lr"] = lr
            #     return lr
            # if param_group["name"] == "t" and self.gaussian_dim == 4:
            #     lr = self.xyz_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_zero(self, reset_value=1.0e-6):
        opacities_new = torch.full_like(self.get_opacity, reset_value)
        opacities_new = inverse_sigmoid(opacities_new)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_high(self):
        opacities = self.get_opacity
        mask = opacities > 0.995
        opacities_new = opacities.clone()
        opacities_new[mask] = 0.99
        opacities_new = inverse_sigmoid(opacities_new)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_specular_high(self):
        specular2 = self.get_specular2
        mask_high = specular2 > 0.995
        mask_low = specular2 < -0.995
        specular2_new = specular2.clone()
        specular2_new[mask_high] = 0.99
        specular2_new[mask_low] = -0.99
        specular2_new = inverse_tanh(specular2_new)
        optimizable_tensors = self.replace_tensor_to_optimizer(specular2_new, "specular2")
        self._specular2 = optimizable_tensors["specular2"]

    def reset_feature(self):
        feature_new = torch.zeros_like(self._specular2)
        optimizable_tensors = self.replace_tensor_to_optimizer(feature_new, "specular2")
        self._specular2 = optimizable_tensors["specular2"]

    def reset_diffuse(self):
        mask = self.get_specular.mean(dim=-1) < 0.8
        #mask2 = self.get_specular2.mean(dim=-1) < 0.8
        diffuse_new = torch.zeros_like(self._features_dc)
        diffuse_new[mask] = self._features_dc[mask]
        optimizable_tensors = self.replace_tensor_to_optimizer(diffuse_new, "f_dc")
        self._features_dc = optimizable_tensors["f_dc"]

    def reset_albedo_from_sh(self, min_albedo=1.0e-4, max_albedo=1.0):
        with torch.no_grad():
            sh_color = eval_sh(0, self._features_dc.transpose(1, 2), None) + 0.5
            target_albedo = torch.clamp(sh_color, min=min_albedo, max=max_albedo)
            albedo_new = torch.log(target_albedo * 5.0)
        if self.optimizer is None:
            self._albedo = nn.Parameter(albedo_new.requires_grad_(True))
            return
        optimizable_tensors = self.replace_tensor_to_optimizer(albedo_new, "albedo")
        if "albedo" in optimizable_tensors:
            self._albedo = optimizable_tensors["albedo"]
        else:
            self._albedo = nn.Parameter(albedo_new.requires_grad_(True))

    def reset_opacity_cpu(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        self._opacity = opacities_new
        self.opt_states["opacity"]["exp_avg"] = torch.zeros_like(self._opacity)
        self.opt_states["opacity"]["exp_avg_sq"] = torch.zeros_like(self._opacity)

    def reset_diffuse_cpu(self):
        diffuse_new = torch.zeros_like(self._features_dc)
        mask = self.get_specular.mean(dim=-1) < 0.8
        self._features_dc[~mask] = diffuse_new[~mask]
        self.opt_states["f_dc"]["exp_avg"] = torch.zeros_like(self._features_dc)
        self.opt_states["f_dc"]["exp_avg_sq"] = torch.zeros_like(self._features_dc)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    del self.optimizer.state[group['params'][0]]
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                if stored_state is not None:
                    self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def get_optimizer_states_by_mask(self, mask, repeat_n=1):
        """Collect per-point Adam moments (exp_avg / exp_avg_sq) for the selected
        points so children created from them can inherit the converged optimizer
        state instead of starting from zeroed moments (which causes large
        sign-normalized first steps and destabilizes freshly split Gaussians)."""
        states = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ("brdf_mlp", "light_mlp", "light_mlp2", "dir_encoding"):
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is None or "exp_avg" not in stored_state:
                continue
            exp_avg = stored_state["exp_avg"][mask]
            exp_avg_sq = stored_state["exp_avg_sq"][mask]
            if repeat_n > 1:
                repeat_dims = [repeat_n] + [1] * (exp_avg.dim() - 1)
                exp_avg = exp_avg.repeat(*repeat_dims)
                exp_avg_sq = exp_avg_sq.repeat(*repeat_dims)
            states[group["name"]] = {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}
        return states

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "brdf_mlp":
                continue
            if group["name"] == "light_mlp":
                continue
            if group["name"] == "light_mlp2":
                continue
            if group["name"] == "dir_encoding":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.specular_time_gradient_accum = self.specular_time_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        if self.gaussian_dim == 4:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            if self.rot_4d:
                self._velocity = optimizable_tensors['velocity']
                self._velocity2 = optimizable_tensors['velocity2']
                self._velocity3 = optimizable_tensors['velocity3']
                self._rot_velocity = optimizable_tensors['rot_velocity']
                self._specular = optimizable_tensors['specular']
                self._albedo = optimizable_tensors['albedo']
                self._specular2 = optimizable_tensors['specular2']
                self._delta_normal = optimizable_tensors['delta_normal']
                self._roughness = optimizable_tensors['roughness']
            self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict, states_dict=None):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "brdf_mlp":
                continue
            if group["name"] == "light_mlp":
                continue
            if group["name"] == "light_mlp2":
                continue
            if group["name"] == "dir_encoding":
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                extension_state = None if states_dict is None else states_dict.get(group["name"], None)
                if extension_state is not None:
                    exp_avg_ext = extension_state["exp_avg"].to(stored_state["exp_avg"].dtype).to(stored_state["exp_avg"].device)
                    exp_avg_sq_ext = extension_state["exp_avg_sq"].to(stored_state["exp_avg_sq"].dtype).to(stored_state["exp_avg_sq"].device)
                else:
                    exp_avg_ext = torch.zeros_like(extension_tensor)
                    exp_avg_sq_ext = torch.zeros_like(extension_tensor)

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], exp_avg_ext), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], exp_avg_sq_ext), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_t, new_scaling_t, new_velocity, new_velocity2, new_velocity3, new_rot_velocity, new_specular, new_albedo, new_specular2, new_delta_normal, new_roughness, new_opt_states=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        }
        if self.gaussian_dim == 4:
            d["t"] = new_t
            d["scaling_t"] = new_scaling_t
            if self.rot_4d:
                d["velocity"] = new_velocity
                d["velocity2"] = new_velocity2
                d["velocity3"] = new_velocity3
                d["rot_velocity"] = new_rot_velocity
                d["specular"] = new_specular
                d["albedo"] = new_albedo
                d["specular2"] = new_specular2
                d["delta_normal"] = new_delta_normal
                d["roughness"] = new_roughness

        optimizable_tensors = self.cat_tensors_to_optimizer(d, states_dict=new_opt_states)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.gaussian_dim == 4:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            if self.rot_4d:
                self._velocity = optimizable_tensors['velocity']
                self._velocity2 = optimizable_tensors['velocity2']
                self._velocity3 = optimizable_tensors['velocity3']
                self._rot_velocity = optimizable_tensors['rot_velocity']
                self._specular = optimizable_tensors['specular']
                self._albedo = optimizable_tensors['albedo']
                self._specular2 = optimizable_tensors['specular2']
                self._delta_normal = optimizable_tensors['delta_normal']
                self._roughness = optimizable_tensors['roughness']
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.specular_time_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def get_outside_msk(self, xyz, ENV_CENTER, ENV_RADIUS):
        if ENV_CENTER is None or ENV_RADIUS is None:
            #print("mask", (torch.zeros(xyz.shape[0], device="cuda", dtype=torch.bool)).size())
            return torch.zeros(xyz.shape[0], device="cuda", dtype=torch.bool)
        #print("mask", (torch.sum((xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2).size())
        return torch.sum((xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2

    def densify_and_split_time(self, grads_t, grads_spec_t, grad_t_threshold, grad_spec_t_threshold, N=2, defer_prune=False, parent_exclusion_mask=None, grad_t_quantile=0.99, grad_t_floor=0.0):
        if grad_spec_t_threshold is None:
            return
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        # padded_grad = torch.zeros((n_init_points), device="cuda")
        # padded_grad[:grads.shape[0]] = grads.squeeze()
        # selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        # spatial_spread_mask = torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        # # if self.gaussian_dim == 4:
        # #     time_span = max(self.time_duration[1] - self.time_duration[0], 1.0e-6)
        # #     temporal_spread_mask = self.get_scaling_t.squeeze(-1) > self.percent_dense * time_span
        # #     spread_mask = torch.logical_or(spatial_spread_mask, temporal_spread_mask)
        # # else:
        # spread_mask = spatial_spread_mask
        selected_pts_mask = torch.zeros((n_init_points), dtype=torch.bool, device="cuda")
        half_frame = 0.5 / 30.0
        flat_radius_mask = self.get_temporal_flat_radius().squeeze(-1) >= half_frame
        # Photometric time-gradient criterion, self-calibrating: select the top 1%
        # of points by mean |dL/dt| each split event (bounded; the flat-radius /
        # temporal-range / inside-sphere gates below still apply).  Lets fast
        # DIFFUSE content (moving leg shadows) trigger temporal splits.
        if self.gaussian_dim == 4 and grads_t is not None and grads_spec_t is not None and grad_spec_t_threshold is not None:
            padded_grad_t = torch.zeros((n_init_points), device="cuda")
            padded_grad_t[:grads_t.shape[0]] = grads_t.squeeze()
            pos_t = padded_grad_t[padded_grad_t > 0]
            if pos_t.numel() > 1000:
                # quantile self-calibrates the working point; the floor stops
                # splits once the population's |dL/dt| tail converges below it
                t_thr = torch.clamp_min(torch.quantile(pos_t, grad_t_quantile), grad_t_floor)
                t_cand = padded_grad_t >= t_thr
                selected_pts_mask = torch.logical_or(selected_pts_mask, t_cand)
                print("time-split t-grad candidates:", int(t_cand.sum()), "thr", float(t_thr))

        if self.gaussian_dim == 4 and grads_spec_t is not None and grad_spec_t_threshold is not None:
            padded_grad_spec_t = torch.zeros((n_init_points), device="cuda")
            padded_grad_spec_t[:grads_spec_t.shape[0]] = grads_spec_t.squeeze()
            selected_pts_mask = torch.logical_or(selected_pts_mask, padded_grad_spec_t >= grad_spec_t_threshold)
            print("max grads_spec_t: ", padded_grad_spec_t.max())
            min_scale_t_mask = self.get_temporal_range_for_opacity(0.05).squeeze(-1) > (1 / 30 / 2)
            #ENV_CENTER = torch.tensor([0, 1, 3], device="cuda")
            ENV_CENTER = torch.tensor([0, 0, 0], device="cuda")
            #ENV_RADIUS = 1.6
            ENV_RADIUS = 8
            #ENV_RADIUS = 2
            xyz = self.get_xyz
            outside_mask = self.get_outside_msk(xyz, ENV_CENTER, ENV_RADIUS)
            gs_in = torch.ones(xyz.shape[0], device="cuda")
            gs_in[outside_mask] = 0.0
            #min_scale_t_mask = torch.sqrt(-2 * torch.log(torch.tensor(0.05, device="cuda")) * self.get_sigma_t).squeeze(-1) > (1 / 30 / 2)
            print("mask size", selected_pts_mask.sum(), flat_radius_mask.sum(), min_scale_t_mask.sum(), gs_in.sum())
            selected_pts_mask = torch.logical_and(selected_pts_mask, flat_radius_mask)
            selected_pts_mask = torch.logical_and(selected_pts_mask, min_scale_t_mask)
            selected_pts_mask = torch.logical_and(selected_pts_mask, gs_in.bool())
        if parent_exclusion_mask is not None:
            exclusion_mask = torch.zeros((n_init_points), dtype=torch.bool, device="cuda")
            mask_count = min(parent_exclusion_mask.shape[0], n_init_points)
            exclusion_mask[:mask_count] = parent_exclusion_mask[:mask_count]
            selected_pts_mask = torch.logical_and(selected_pts_mask, ~exclusion_mask)
        # print(f"num_to_densify_pos: {torch.where(padded_grad >= grad_threshold, True, False).sum()}, num_to_split_pos: {selected_pts_mask.sum()}")
        print("densify_and_split_time: ", selected_pts_mask.sum())
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        
        # if not self.rot_4d:
        #     stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        #     means = torch.zeros((stds.size(0), 3),device=self.device)
        #     samples = torch.normal(mean=means, std=stds)
        #     rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        #     new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        #     new_t = None
        #     new_scaling_t = None
        #     new_velocity = None
        #     new_velocity2 = None
        #     new_velocity3 = None
        #     new_rot_velocity = None
        #     new_specular = None
        #     new_specular2 = None
        #     new_delta_normal = None
        #     new_roughness = None
        #     if self.gaussian_dim == 4:
        #         stds_t = self.get_scaling_t[selected_pts_mask].repeat(N,1)
        #         means_t = torch.zeros((stds_t.size(0), 1),device=self.device)
        #         samples_t = torch.normal(mean=means_t, std=stds_t)
        #         new_t = samples_t + self.get_t[selected_pts_mask].repeat(N, 1)
        #         new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
        # else:
        stds = self.get_scaling_xyzt[selected_pts_mask].repeat(N,1)
        stds = stds[:,0:3]
        means = torch.zeros((stds.size(0), 3),device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_t, new_scaling_t, time_offsets, child_scaling_t = self.get_temporal_split_params(selected_pts_mask, N)
        if child_scaling_t.numel() > 0:
            print("max scaling_t before split: ", self.get_scaling_t[selected_pts_mask].max())
            print("max scaling_t after split: ", child_scaling_t.max())
        parent_xyz = self.get_xyz[selected_pts_mask].repeat(N, 1)
        parent_velocity = self._velocity[selected_pts_mask].repeat(N, 1)
        parent_sigma_t_fixed = self.get_sigma_t_fixed[selected_pts_mask]
        new_xyz = parent_xyz + time_offsets * parent_velocity / parent_sigma_t_fixed.repeat(N, 1)
        #new_velocity = torch.zeros_like(self._velocity[selected_pts_mask].repeat(N,1))
        #new_velocity2 = torch.zeros_like(self._velocity2[selected_pts_mask].repeat(N,1))
        new_velocity2 = self._velocity2[selected_pts_mask].repeat(N,1)
        if getattr(self, "temporal_opacity_mode", "normalized_sigmoid") == "flat_window":
            child_flat_radius = self.get_temporal_flat_radius()[selected_pts_mask] / float(N)
            new_velocity2[..., 0:1] = self.temporal_flat_radius_inverse_activation(child_flat_radius).repeat(N, 1)
            #child_sigma_t_fixed = torch.clip(child_scaling_t ** 2 + child_flat_radius, min=1.0)
            child_sigma_t_fixed = 1.0
        else:
            #child_sigma_t_fixed = torch.clip(child_scaling_t ** 2, min=1.0)
            child_sigma_t_fixed = 1.0
        new_velocity3 = torch.zeros_like(self._velocity3[selected_pts_mask].repeat(N,1))
        new_rot_velocity = torch.zeros_like(self._rot_velocity[selected_pts_mask].repeat(N, 1))
        # new_specular = self._specular[selected_pts_mask].repeat(N,1)
        new_albedo = self._albedo[selected_pts_mask].repeat(N,1)
        
        #noise_spec2 = torch.randn_like(self._specular2[selected_pts_mask]) * 0.1
        noise_spec = torch.randn_like(self._specular[selected_pts_mask]) * 0.01
        new_specular_before = self._specular[selected_pts_mask]
        new_specular_after = self._specular[selected_pts_mask]
        new_specular = torch.cat((new_specular_before, new_specular_after), dim=0)
        # new_specular2_before = self._specular2[selected_pts_mask] + noise_spec2
        # new_specular2_after = self._specular2[selected_pts_mask] - noise_spec2
        new_specular2_before = self._specular2[selected_pts_mask]
        new_specular2_after = self._specular2[selected_pts_mask]
        new_specular2 = torch.cat((new_specular2_before, new_specular2_after), dim=0)
        
        new_delta_normal = self._delta_normal[selected_pts_mask].repeat(N,1)
        new_roughness = self._roughness[selected_pts_mask].repeat(N,1)
        #new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1))
        new_velocity = (self._velocity[selected_pts_mask] * child_sigma_t_fixed / parent_sigma_t_fixed).repeat(N,1)
        # new_velocity2 = self._velocity2[selected_pts_mask].repeat(N,1)
        # new_velocity3 = self._velocity3[selected_pts_mask].repeat(N,1)
        #new_rot_velocity = self._rot_velocity[selected_pts_mask].repeat(N, 1)

        # Children inherit the parents' Adam moments: zeroed moments make Adam
        # take ~lr-sized sign-normalized steps on a large, spatially concentrated
        # set of dynamic Gaussians, which breaks the xyz/velocity/t cancellation
        # the split relies on and causes ghosting right after each split round.
        new_opt_states = self.get_optimizer_states_by_mask(selected_pts_mask, repeat_n=N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_t, new_scaling_t, new_velocity, new_velocity2, new_velocity3, new_rot_velocity, new_specular, new_albedo, new_specular2, new_delta_normal, new_roughness, new_opt_states=new_opt_states)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        if defer_prune:
            return prune_filter
        self.prune_points(prune_filter)
        return None

    def densify_and_split_time3(self, grads_t, grads_spec_t, grad_t_threshold, grad_spec_t_threshold, N=3):
        if grad_spec_t_threshold is None:
            return
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        # padded_grad = torch.zeros((n_init_points), device="cuda")
        # padded_grad[:grads.shape[0]] = grads.squeeze()
        # selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        # spatial_spread_mask = torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        # # if self.gaussian_dim == 4:
        # #     time_span = max(self.time_duration[1] - self.time_duration[0], 1.0e-6)
        # #     temporal_spread_mask = self.get_scaling_t.squeeze(-1) > self.percent_dense * time_span
        # #     spread_mask = torch.logical_or(spatial_spread_mask, temporal_spread_mask)
        # # else:
        # spread_mask = spatial_spread_mask
        selected_pts_mask = torch.zeros((n_init_points), dtype=torch.bool, device="cuda")
        if self.gaussian_dim == 4 and grads_t is not None and grad_t_threshold is not None and grads_spec_t is not None and grad_spec_t_threshold is not None:
            padded_grad_t = torch.zeros((n_init_points), device="cuda")
            padded_grad_t[:grads_t.shape[0]] = grads_t.squeeze()
            selected_pts_mask = torch.logical_or(selected_pts_mask, padded_grad_t >= grad_t_threshold)

        if self.gaussian_dim == 4 and grads_spec_t is not None and grad_spec_t_threshold is not None:
            padded_grad_spec_t = torch.zeros((n_init_points), device="cuda")
            padded_grad_spec_t[:grads_spec_t.shape[0]] = grads_spec_t.squeeze()
            selected_pts_mask = torch.logical_or(selected_pts_mask, padded_grad_spec_t >= grad_spec_t_threshold)
            print("max grads_spec_t: ", padded_grad_spec_t.max())
            min_scale_t_mask = self.get_temporal_range_for_opacity(0.05).squeeze(-1) > (1 / 30 / 2)
            print("mask size", selected_pts_mask.sum(), min_scale_t_mask.sum())
            selected_pts_mask = torch.logical_and(selected_pts_mask, min_scale_t_mask)
        # print(f"num_to_densify_pos: {torch.where(padded_grad >= grad_threshold, True, False).sum()}, num_to_split_pos: {selected_pts_mask.sum()}")
        print("densify_and_split_time: ", selected_pts_mask.sum())
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        
        # if not self.rot_4d:
        #     stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        #     means = torch.zeros((stds.size(0), 3),device=self.device)
        #     samples = torch.normal(mean=means, std=stds)
        #     rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        #     new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        #     new_t = None
        #     new_scaling_t = None
        #     new_velocity = None
        #     new_velocity2 = None
        #     new_velocity3 = None
        #     new_rot_velocity = None
        #     new_specular = None
        #     new_specular2 = None
        #     new_delta_normal = None
        #     new_roughness = None
        #     if self.gaussian_dim == 4:
        #         stds_t = self.get_scaling_t[selected_pts_mask].repeat(N,1)
        #         means_t = torch.zeros((stds_t.size(0), 1),device=self.device)
        #         samples_t = torch.normal(mean=means_t, std=stds_t)
        #         new_t = samples_t + self.get_t[selected_pts_mask].repeat(N, 1)
        #         new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
        # else:
        stds = self.get_scaling_xyzt[selected_pts_mask].repeat(N,1)
        stds = stds[:,0:3]
        means = torch.zeros((stds.size(0), 3),device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_t, new_scaling_t, time_offsets, child_scaling_t = self.get_temporal_split_params(selected_pts_mask, N)
        parent_xyz = self.get_xyz[selected_pts_mask].repeat(N, 1)
        parent_velocity = self._velocity[selected_pts_mask].repeat(N, 1)
        parent_sigma_t_fixed = self.get_sigma_t_fixed[selected_pts_mask]
        new_xyz = parent_xyz + time_offsets * parent_velocity / parent_sigma_t_fixed.repeat(N, 1)
        #new_velocity = torch.zeros_like(self._velocity[selected_pts_mask].repeat(N,1))
        new_velocity2 = self._velocity2[selected_pts_mask].repeat(N,1)
        if getattr(self, "temporal_opacity_mode", "normalized_sigmoid") == "flat_window":
            child_flat_radius = self.get_temporal_flat_radius()[selected_pts_mask] / float(N)
            new_velocity2[..., 0:1] = self.temporal_flat_radius_inverse_activation(child_flat_radius).repeat(N, 1)
            #child_sigma_t_fixed = torch.clip(child_scaling_t ** 2 + child_flat_radius, min=1.0)
            child_sigma_t_fixed = 1.0
        else:
            #child_sigma_t_fixed = torch.clip(child_scaling_t ** 2, min=1.0)
            child_sigma_t_fixed = 1.0
        new_velocity3 = torch.zeros_like(self._velocity3[selected_pts_mask].repeat(N,1))
        new_rot_velocity = torch.zeros_like(self._rot_velocity[selected_pts_mask].repeat(N, 1))
        new_specular = self._specular[selected_pts_mask].repeat(N,1)
        new_albedo = self._albedo[selected_pts_mask].repeat(N,1)
        new_specular2 = self._specular2[selected_pts_mask].repeat(N,1)
        new_delta_normal = self._delta_normal[selected_pts_mask].repeat(N,1)
        new_roughness = self._roughness[selected_pts_mask].repeat(N,1)
        #new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1))
        new_velocity = (self._velocity[selected_pts_mask] * child_sigma_t_fixed / parent_sigma_t_fixed).repeat(N,1)
        # new_velocity2 = self._velocity2[selected_pts_mask].repeat(N,1)
        # new_velocity3 = self._velocity3[selected_pts_mask].repeat(N,1)
        #new_rot_velocity = self._rot_velocity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_t, new_scaling_t, new_velocity, new_velocity2, new_velocity3, new_rot_velocity, new_specular, new_albedo, new_specular2, new_delta_normal, new_roughness)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_split(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold, inside_mask, outside_mask, N=2, defer_prune=False):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        
        padded_inside_mask = torch.zeros((n_init_points), device="cuda", dtype=torch.bool)
        padded_inside_mask[:inside_mask.shape[0]] = inside_mask
        
        padded_outside_mask = torch.zeros((n_init_points), device="cuda", dtype=torch.bool)
        padded_outside_mask[:outside_mask.shape[0]] = outside_mask

        #selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_or(torch.logical_and(torch.where(padded_grad >= grad_threshold, True, False), padded_inside_mask), torch.logical_and(torch.where(padded_grad >= grad_threshold, True, False), padded_outside_mask))

        spatial_spread_mask = torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        # if self.gaussian_dim == 4:
        #     time_span = max(self.time_duration[1] - self.time_duration[0], 1.0e-6)
        #     temporal_spread_mask = self.get_scaling_t.squeeze(-1) > self.percent_dense * time_span
        #     spread_mask = torch.logical_or(spatial_spread_mask, temporal_spread_mask)
        # else:
        spread_mask = spatial_spread_mask

        # if self.gaussian_dim == 4 and grads_t is not None and grad_t_threshold is not None and grads_spec_t is not None and grad_spec_t_threshold is not None:
        #     padded_grad_t = torch.zeros((n_init_points), device="cuda")
        #     padded_grad_t[:grads_t.shape[0]] = grads_t.squeeze()
        #     selected_pts_mask = torch.logical_or(selected_pts_mask, padded_grad_t >= grad_t_threshold)

        # if self.gaussian_dim == 4 and grads_spec_t is not None and grad_spec_t_threshold is not None:
        #     padded_grad_spec_t = torch.zeros((n_init_points), device="cuda")
        #     padded_grad_spec_t[:grads_spec_t.shape[0]] = grads_spec_t.squeeze()
        #     selected_pts_mask = torch.logical_or(selected_pts_mask, padded_grad_spec_t >= grad_spec_t_threshold)

        selected_pts_mask = torch.logical_and(selected_pts_mask, spread_mask)
        # print(f"num_to_densify_pos: {torch.where(padded_grad >= grad_threshold, True, False).sum()}, num_to_split_pos: {selected_pts_mask.sum()}")
        
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        
        if not self.rot_4d:
            stds = self.get_scaling[selected_pts_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 3),device=self.device)
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_t = None
            new_scaling_t = None
            new_velocity = None
            new_velocity2 = None
            new_velocity3 = None
            new_rot_velocity = None
            new_specular = None
            new_albedo = None
            new_specular2 = None
            new_delta_normal = None
            new_roughness = None
            if self.gaussian_dim == 4:
                stds_t = self.get_scaling_t[selected_pts_mask].repeat(N,1)
                means_t = torch.zeros((stds_t.size(0), 1),device=self.device)
                samples_t = torch.normal(mean=means_t, std=stds_t)
                new_t = samples_t + self.get_t[selected_pts_mask].repeat(N, 1)
                new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
        else:
            stds = self.get_scaling_xyzt[selected_pts_mask].repeat(N,1)
            stds = stds[:,0:3]
            means = torch.zeros((stds.size(0), 3),device=self.device)
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
            new_xyzt = self.get_xyzt[selected_pts_mask].repeat(N, 1)# + torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            #new_xyz = new_xyzt[...,0:3]
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_t = new_xyzt[...,3:4]
            #new_scaling_t = torch.zeros_like(self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1))) + torch.log(torch.tensor(0.4 * 20))
            new_scaling_t = self._scaling_t[selected_pts_mask].repeat(N,1)
            #new_velocity = torch.zeros_like(self._velocity[selected_pts_mask].repeat(N,1))
            #new_velocity2 = torch.zeros_like(self._velocity2[selected_pts_mask].repeat(N,1))
            new_velocity2 = self._velocity2[selected_pts_mask].repeat(N,1)
            new_velocity3 = torch.zeros_like(self._velocity3[selected_pts_mask].repeat(N,1))
            new_rot_velocity = torch.zeros_like(self._rot_velocity[selected_pts_mask].repeat(N, 1))
            new_specular = self._specular[selected_pts_mask].repeat(N,1)
            new_albedo = self._albedo[selected_pts_mask].repeat(N,1)
            # noise_spec2 = torch.randn_like(self._specular2[selected_pts_mask]) * 0.1
            # new_specular2 = torch.cat((self._specular2[selected_pts_mask] + noise_spec2, self._specular2[selected_pts_mask] - noise_spec2), dim=0)
            new_specular2 = self._specular2[selected_pts_mask].repeat(N,1)
            new_delta_normal = self._delta_normal[selected_pts_mask].repeat(N,1)
            new_roughness = self._roughness[selected_pts_mask].repeat(N,1)
            #new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1))
            new_velocity = self._velocity[selected_pts_mask].repeat(N,1)
            # new_velocity2 = self._velocity2[selected_pts_mask].repeat(N,1)
            # new_velocity3 = self._velocity3[selected_pts_mask].repeat(N,1)
            #new_rot_velocity = self._rot_velocity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_t, new_scaling_t, new_velocity, new_velocity2, new_velocity3, new_rot_velocity, new_specular, new_albedo, new_specular2, new_delta_normal, new_roughness)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        if defer_prune:
            return prune_filter
        self.prune_points(prune_filter)
        return None

    def densify_and_clone(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold, inside_mask, outside_mask, low_opa=True):
        # Extract points that satisfy the gradient condition
        #selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_or(torch.logical_and(torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False), inside_mask), torch.logical_and(torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False), outside_mask))
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # print(f"num_to_densify_pos: {torch.where(grads >= grad_threshold, True, False).sum()}, num_to_clone_pos: {selected_pts_mask.sum()}")
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        if low_opa:
            new_opacities = self.inverse_opacity_activation(1 - (1 - self.opacity_activation(new_opacities)) ** 0.5)
            # KNOWN NO-OP, kept deliberately: this assigns into an advanced-indexing
            # temporary, so the parent keeps full opacity while the child gets the
            # reduced value. Fixing it to `self._opacity.data[mask] = ...` was tested
            # twice (exp73 bundle -0.05, exp82 single-variable -0.17 vs exp78) and
            # lost both times — parents staying opaque during pre-9k clones is
            # load-bearing for early densification dynamics.
            self._opacity[selected_pts_mask].data = new_opacities
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_t = None
        new_scaling_t = None
        new_velocity = None
        new_velocity2 = None
        new_velocity3 = None
        new_rot_velocity = None
        new_specular = None
        new_albedo = None
        new_specular2 = None
        new_delta_normal = None
        new_roughness = None
        if self.gaussian_dim == 4:
            new_t = self._t[selected_pts_mask]
            #new_scaling_t = torch.zeros_like(self._scaling_t[selected_pts_mask]) + torch.log(torch.tensor(0.4 * 20))
            new_scaling_t = self._scaling_t[selected_pts_mask]
            if self.rot_4d:
                #new_velocity = torch.zeros_like(self._velocity[selected_pts_mask])
                #new_velocity2 = torch.zeros_like(self._velocity2[selected_pts_mask])
                new_velocity2 = self._velocity2[selected_pts_mask]
                new_velocity3 = torch.zeros_like(self._velocity3[selected_pts_mask])
                new_rot_velocity = torch.zeros_like(self._rot_velocity[selected_pts_mask])
                new_specular = self._specular[selected_pts_mask]
                new_albedo = self._albedo[selected_pts_mask]
                # noise_spec2 = torch.randn_like(self._specular2[selected_pts_mask]) * 0.1
                # new_specular2 = self._specular2[selected_pts_mask] + noise_spec2
                new_specular2 = self._specular2[selected_pts_mask]
                new_delta_normal = self._delta_normal[selected_pts_mask]
                new_roughness = self._roughness[selected_pts_mask]
                new_velocity = self._velocity[selected_pts_mask]
                # new_velocity2 = self._velocity2[selected_pts_mask]
                # new_velocity3 = self._velocity3[selected_pts_mask]
                # new_rot_velocity = self._rot_velocity[selected_pts_mask]

            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_t, new_scaling_t, new_velocity, new_velocity2, new_velocity3, new_rot_velocity, new_specular, new_albedo, new_specular2, new_delta_normal, new_roughness)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration, max_grad_t=None, max_specular_time_grad=None, prune_only=False, disable_prune=False, split_time=False, grad_t_quantile=0.99, grad_t_floor=0.0):
        # Every per-gaussian parameter gets replaced by a new nn.Parameter below
        # (grad=None), so the optimizer step that follows can never consume the
        # old .grad buffers; dropping them first removes a params-worth of
        # memory from the surgery peak.  Module groups keep theirs.
        for group in self.optimizer.param_groups:
            if group["name"] in ("brdf_mlp", "light_mlp", "light_mlp2", "dir_encoding"):
                continue
            for param in group["params"]:
                param.grad = None
        #ENV_CENTER = torch.tensor([0, 1, 3], device="cuda")
        ENV_CENTER = torch.tensor([0, 0, 0], device="cuda")
        #ENV_RADIUS = 1.6
        ENV_RADIUS = 8
        #ENV_RADIUS = 2
        xyz = self.get_xyz
        outside_mask = self.get_outside_msk(xyz, ENV_CENTER, ENV_RADIUS)
        gs_in = torch.ones(xyz.shape[0], device="cuda", dtype=torch.bool)
        gs_in[outside_mask] = False
        
        # n_init_points = self.get_xyz.shape[0]
        # padded_inside_mask = torch.zeros((n_init_points), device="cuda", dtype=torch.bool)
        # padded_inside_mask[:gs_in.shape[0]] = gs_in
        
        # padded_outside_mask = torch.zeros((n_init_points), device="cuda", dtype=torch.bool)
        # padded_outside_mask[:outside_mask.shape[0]] = outside_mask

        deferred_prune_filter = None

        def merge_prune_filters(*filters):
            valid_filters = [mask for mask in filters if mask is not None]
            if not valid_filters:
                return None
            merged = torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.bool)
            for mask in valid_filters:
                mask_count = min(mask.shape[0], merged.shape[0])
                merged[:mask_count] = torch.logical_or(merged[:mask_count], mask[:mask_count])
            return merged

        if not prune_only:
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            grads_abs = self.xyz_gradient_accum_abs / self.denom
            grads_abs[grads_abs.isnan()] = 0.0
            if self.gaussian_dim == 4:
                grads_t = self.t_gradient_accum / self.denom
                grads_t[grads_t.isnan()] = 0.0
                grads_spec_t = None
                if max_specular_time_grad is not None and max_grad_t is not None and max_specular_time_grad > 0:
                    grads_spec_t = self.specular_time_gradient_accum
                    grads_spec_t[grads_spec_t.isnan()] = 0.0
                # grads_spec_t = None
                # if max_specular_time_grad is not None and max_grad_t is not None and max_specular_time_grad > 0:
                #     grads_spec_t = self.specular_time_gradient_accum / self.denom
                #     grads_spec_t[grads_spec_t.isnan()] = 0.0
                    # grads_t_norm = grads_t / (max_grad_t + 1.0e-12)
                    # grads_spec_t_norm = grads_spec_t / (max_specular_time_grad + 1.0e-12)
                    # grads_t = torch.maximum(grads_t_norm, grads_spec_t_norm)
                    # max_grad_t = 1.0
            else:
                grads_t = None
            spatial_prune_filter = None
            temporal_prune_filter = None
            if not split_time:
                if iteration < 9000:
                    self.densify_and_clone(grads, max_grad, extent, grads_t, max_grad_t, gs_in, outside_mask)
                else:
                    self.densify_and_clone(grads, max_grad, extent, grads_t, max_grad_t, gs_in, outside_mask, low_opa=False)
                spatial_prune_filter = self.densify_and_split(grads_abs, max_grad * 2, extent, grads_t, max_grad_t, gs_in, outside_mask, defer_prune=True)
            #self.densify_and_split_time3(grads_t, grads_spec_t, max_grad_t, max_specular_time_grad)
            else:
                temporal_prune_filter = self.densify_and_split_time(grads_t, grads_spec_t, max_grad_t, max_specular_time_grad, defer_prune=True, grad_t_quantile=grad_t_quantile, grad_t_floor=grad_t_floor)
            deferred_prune_filter = merge_prune_filters(spatial_prune_filter, temporal_prune_filter)

        #prune_mask = (self.get_opacity < min_opacity).squeeze()
        if not disable_prune:
            n_init_points = self.get_xyz.shape[0]
            padded_inside_mask = torch.zeros((n_init_points), device="cuda", dtype=torch.bool)
            padded_inside_mask[:gs_in.shape[0]] = gs_in

            padded_outside_mask = torch.zeros((n_init_points), device="cuda", dtype=torch.bool)
            padded_outside_mask[:outside_mask.shape[0]] = outside_mask
            prune_mask = torch.logical_or(torch.logical_and((self.get_opacity < min_opacity).squeeze(), padded_inside_mask), torch.logical_and((self.get_opacity < 0.05).squeeze(), padded_outside_mask))
            if max_screen_size:
                big_points_vs = self.max_radii2D > max_screen_size
                big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        else:
            prune_mask = torch.zeros_like(self.get_opacity.squeeze(), dtype=torch.bool)
        deferred_prune_filter = merge_prune_filters(deferred_prune_filter)
        if deferred_prune_filter is not None:
            prune_mask = torch.logical_or(prune_mask, deferred_prune_filter)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def densify_and_prune_time(self, min_opacity, extent, max_screen_size, max_grad_t=None, max_specular_time_grad=None, prune_only=False):
        if not prune_only:
            # grads = self.xyz_gradient_accum / self.denom
            # grads[grads.isnan()] = 0.0
            # grads_abs = self.xyz_gradient_accum_abs / self.denom
            # grads_abs[grads_abs.isnan()] = 0.0
            if self.gaussian_dim == 4:
                grads_t = self.t_gradient_accum / self.denom
                grads_t[grads_t.isnan()] = 0.0
                grads_spec_t = None
                if max_specular_time_grad is not None and max_grad_t is not None and max_specular_time_grad > 0:
                    grads_spec_t = self.specular_time_gradient_accum
                    grads_spec_t[grads_spec_t.isnan()] = 0.0
                    # grads_t_norm = grads_t / (max_grad_t + 1.0e-12)
                    # grads_spec_t_norm = grads_spec_t / (max_specular_time_grad + 1.0e-12)
                    # grads_t = torch.maximum(grads_t_norm, grads_spec_t_norm)
                    # max_grad_t = 1.0
            else:
                grads_t = None

            # self.densify_and_clone(grads, max_grad, extent, grads_t, max_grad_t)
            # self.densify_and_split(grads_abs, max_grad * 2, extent, grads_t, grads_spec_t, max_grad_t, max_specular_time_grad)
            self.densify_and_split_time(grads_t, grads_spec_t, max_grad_t, max_specular_time_grad)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, avg_t_grad=None):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:4], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]
            if self.rot_4d and self._specular2.numel() > 0:
                self.specular_time_gradient_accum[update_filter] = torch.max(self.specular_time_gradient_accum[update_filter], self.get_specular2_temporal_variation[update_filter])
        
    def add_densification_stats_pgsr(self, viewspace_point_tensor, viewspace_point_tensor_abs, update_filter, avg_t_grad=None, add_specular_time_grad=False):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor_abs.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4:
            # ABS accumulation: signed t-grads cancel across frames (same pathology
            # AbsGS fixed for xyz), hiding temporally-misfit diffuse content.
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter].abs()
            if add_specular_time_grad and self.rot_4d and self._specular2.numel() > 0:
                self.specular_time_gradient_accum[update_filter] = torch.max(self.specular_time_gradient_accum[update_filter], self.get_specular2_temporal_variation[update_filter])
        
    def add_densification_stats_grad(self, viewspace_point_grad, update_filter, avg_t_grad=None):
        self.xyz_gradient_accum[update_filter] += viewspace_point_grad[update_filter]
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]
            if self.rot_4d and self._specular2.numel() > 0:
                self.specular_time_gradient_accum[update_filter] = torch.max(self.specular_time_gradient_accum[update_filter], self.get_specular2_temporal_variation[update_filter])

    def set_current_timestamp(self, current_timestamp : float):
        self.current_timestamp = current_timestamp
        
class AdamWithMaskedUpdates(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, mask=None, **kwargs):
        super(AdamWithMaskedUpdates, self).__init__(params, lr=lr, **kwargs)

    def step(self, closure=None):
        # 计算损失
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                # p.data = torch.where(p.data.isnan(), torch.zeros_like(p.data), p.data)  # 处理 NaN 值
                # if 'rot' in group['name']:
                #     p.data[..., 0] = 1
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state.get(p, None)
                if state is None:
                    self.state[p] = {}
                    state = self.state[p]
                    
                if 'exp_avg' not in state:
                    state['step'] = torch.tensor(0, dtype=torch.int, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = self.defaults['betas']
                state['step'] += 1

                # 更新动量项，仅在 mask 为 1 的位置更新
                exp_avg.mul_(beta1)  # 仅更新 mask 为 1 的位置
                exp_avg.add_(grad, alpha=(1 - beta1))
                # exp_avg *= beta1  # 仅更新 mask 为 1 的位置
                # print(exp_avg.shape, grad.shape, (1 - beta1), group['name'])
                # exp_avg += grad * (1 - beta1)
                exp_avg_sq.mul_(beta2)  # 仅更新 mask 为 1 的位置
                exp_avg_sq.addcmul_(grad, grad, value=(1 - beta2))
                # exp_avg_sq *= beta2  # 仅更新 mask 为 1 的位置
                # exp_avg_sq += grad * grad * (1 - beta2)

                # 计算偏差修正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = lr / bias_correction1

                denom = exp_avg_sq.sqrt() / bias_correction2 + self.defaults['eps']
                p.data -= step_size * exp_avg / denom
                # if p.data.isnan().any():
                #     print(f"NaN detected in parameter {group['name']}, resetting to zero")
                # state['exp_avg'] = torch.where(p.data.isnan(), torch.zeros_like(p.data), state['exp_avg'])  # 处理 NaN 值
                # state['exp_avg_sq'] = torch.where(p.data.isnan(), torch.zeros_like(p.data), state['exp_avg_sq'])  # 处理 NaN 值
                # p.data = torch.where(p.data.isnan(), torch.zeros_like(p.data), p.data)  # 处理 NaN 值
                # # if 'scal' in group['name']:
                # #     p.data[..., :] = 0.01
                # if 'rot' in group['name']:
                #     p.data[..., 0] = 1
