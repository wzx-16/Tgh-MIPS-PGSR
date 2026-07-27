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

import torch
from scene import Scene, TemperalGaussianHierarchy
import os
import sys
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_3d_pgsr_anti
import torchvision
from utils.general_utils import safe_state, safe_normalize, reflect, print_tensor_distribution
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import math
from torchvision import transforms
from transformers import pipeline as pp
import numpy as np
import cv2

def _clone_tensor_attr(src_tensor):
    cloned = src_tensor.detach().clone()
    if isinstance(src_tensor, torch.nn.Parameter):
        return torch.nn.Parameter(cloned.requires_grad_(True))
    return cloned


def _explicit_cli_dests(parser):
    """Set of argparse dests explicitly present on the command line."""
    tokens = set()
    for tok in sys.argv[1:]:
        if tok.startswith("-"):
            tokens.add(tok.split("=")[0])
    dests = set()
    for action in parser._actions:
        if any(opt in tokens for opt in action.option_strings):
            dests.add(action.dest)
    return dests


def _apply_render_config(args, parser):
    """Merge a training yaml (or an effective_config_*.yaml dump) onto args.

    Priority: explicit CLI flags > config yaml > cfg_args > argparse defaults.
    Keys unknown to the render parser (train-only knobs) are collected onto
    args.extra_cfg instead of being dropped, so callers can consume e.g.
    local_feature_start_iter.
    """
    args.extra_cfg = {}
    if not getattr(args, "config", None):
        return args
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if isinstance(cfg.get("args"), dict) and "run_id" in cfg:
        cfg = cfg["args"]  # effective_config_*.yaml dump: values live under args:
    flat = {}
    def _walk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                _walk(v)  # ModelParams:/PipelineParams:/OptimizationParams: sections
            else:
                flat[k] = v
    _walk(cfg)
    explicit = _explicit_cli_dests(parser)
    applied, kept_cli = [], []
    for k, v in flat.items():
        if k in ("config",):
            continue
        if k in explicit:
            kept_cli.append(k)
        elif hasattr(args, k):
            setattr(args, k, v)
            applied.append(k)
        else:
            args.extra_cfg[k] = v
    print(f"Applied {len(applied)} params from config {args.config}")
    if kept_cli:
        print("Kept explicit CLI overrides:", sorted(kept_cli))
    return args


def _infer_checkpoint_id(loaded_pth):
    if not loaded_pth:
        return 0
    name = os.path.basename(loaded_pth)
    for prefix in ("tgh", "gaussian", "local"):
        if not name.startswith(prefix):
            continue
        digits = []
        for ch in name[len(prefix):]:
            if not ch.isdigit():
                break
            digits.append(ch)
        if digits:
            return int("".join(digits))
    return 0


def _parse_render_frames(render_frames):
    if not render_frames:
        return None
    frames = set()
    for part in render_frames.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            frames.update(range(int(start), int(end) + 1))
        else:
            frames.add(int(part))
    return frames


def _frame_from_image_name(image_name):
    stem = os.path.splitext(os.path.basename(str(image_name)))[0]
    token = stem.rsplit("_", 1)[-1]
    try:
        return int(token)
    except ValueError:
        return None


def _filter_views_by_frames(views, frame_filter, split_name):
    if frame_filter is None:
        return views
    selected = []
    for view in views:
        frame = _frame_from_image_name(view[2].image_name)
        if frame in frame_filter:
            selected.append(view)
    print(f"Filtered {split_name} views by frames {sorted(frame_filter)}: {len(selected)} / {len(views)}")
    return selected

# def print_tensor_distribution(name, tensor, bins=10):
#     with torch.no_grad():
#         values = tensor.detach().reshape(-1).float()
#         total_count = values.numel()
#         if total_count == 0:
#             print(f"{name} distribution: empty")
#             return

#         finite_values = values[torch.isfinite(values)]
#         finite_count = finite_values.numel()
#         if finite_count == 0:
#             print(f"{name} distribution: total={total_count}, finite=0")
#             return

#         quantiles = torch.quantile(
#             finite_values,
#             torch.tensor([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0], device=finite_values.device),
#         ).detach().cpu().tolist()
#         mean = finite_values.mean().item()
#         std = finite_values.std(unbiased=False).item()

#         min_value = quantiles[0]
#         max_value = quantiles[-1]
#         if min_value == max_value:
#             hist_counts = [finite_count]
#             hist_edges = [min_value, max_value]
#         else:
#             hist_counts = torch.histc(finite_values, bins=bins, min=min_value, max=max_value).detach().cpu().int().tolist()
#             hist_edges = torch.linspace(min_value, max_value, bins + 1).tolist()

#         print(
#             f"{name} distribution: total={total_count}, finite={finite_count}, "
#             f"mean={mean:.6g}, std={std:.6g}, "
#             f"min/p01/p05/median/p95/p99/max={[round(value, 6) for value in quantiles]}, "
#             f"hist_edges={[round(value, 6) for value in hist_edges]}, hist_counts={hist_counts}"
#         )

def initialize_local_gaussian_model(global_model: GaussianModel, local_model: GaussianModel, zero_local_feature=False):
    local_model.gsdim = global_model.gsdim
    local_model.active_sh_degree = global_model.active_sh_degree
    local_model.active_sh_degree_t = global_model.active_sh_degree_t
    local_model.max_sh_degree = global_model.max_sh_degree
    local_model.max_sh_degree_t = global_model.max_sh_degree_t
    local_model.spatial_lr_scale = global_model.spatial_lr_scale
    local_model.current_timestamp = global_model.current_timestamp
    local_model.rot_4d = global_model.rot_4d
    local_model.gaussian_dim = global_model.gaussian_dim
    local_model.force_sh_3d = global_model.force_sh_3d
    local_model.time_duration = global_model.time_duration
    local_model.temporal_opacity_mode = global_model.temporal_opacity_mode
    _topa_override = str(getattr(local_model, "temporal_opacity_mode_override", "") or "")
    if _topa_override:
        local_model.temporal_opacity_mode = _topa_override
    local_model.temporal_flat_radius_mult = global_model.temporal_flat_radius_mult
    local_model.temporal_flat_edge_sigma_mult = global_model.temporal_flat_edge_sigma_mult

    tensor_attrs = [
        "_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity",
        "_t", "_scaling_t", "_velocity", "_velocity2", "_velocity3", "_rot_velocity",
        "_specular", "_albedo", "_specular2", "_delta_normal", "_roughness",
    ]
    for attr in tensor_attrs:
        src_val = getattr(global_model, attr, None)
        if isinstance(src_val, torch.Tensor):
            setattr(local_model, attr, _clone_tensor_attr(src_val))

    if isinstance(global_model.max_radii2D, torch.Tensor):
        local_model.max_radii2D = global_model.max_radii2D.detach().clone()

    if zero_local_feature:
        for attr in ("_specular", "_specular2"):
            value = getattr(local_model, attr, None)
            if isinstance(value, torch.Tensor):
                with torch.no_grad():
                    value.zero_()

def render_set(model_path, name, iteration, views, gaussians, local_gaussians, tgh, pipeline, background, skip_save, id, use_tgh=True, render_suffix="", append_gt=False):
    output_tag = "ours_{}{}".format(iteration, render_suffix)
    render_path = os.path.join(model_path, f"{name}_{id}", output_tag, "renders")
    gts_path = os.path.join(model_path, f"{name}_{id}", output_tag, "gt")
    predicted_depth_path = os.path.join(model_path, f"{name}_{id}", output_tag, "predicted_depth")
    depth_normal_path = os.path.join(model_path, f"{name}_{id}", output_tag, "depth_normal")
    rendered_normal_path = os.path.join(model_path, f"{name}_{id}", output_tag, "rendered_normal")
    depth_path = os.path.join(model_path, f"{name}_{id}", output_tag, "rendered_depth")
    feature_path = os.path.join(model_path, f"{name}_{id}", output_tag, "rendered_feature")
    spec_path = os.path.join(model_path, f"{name}_{id}", output_tag, "rendered_specular")
    alpha_path = os.path.join(model_path, f"{name}_{id}", output_tag, "rendered_alpha")
    in_path = os.path.join(model_path, f"{name}_{id}", output_tag, "rendered_in")
    error_path = os.path.join(model_path, f"{name}_{id}", output_tag, "error")
    delta_normal_path = os.path.join(model_path, f"{name}_{id}", output_tag, "rendered_delta_normal")
    diffuse_path = os.path.join(model_path, f"{name}_{id}", output_tag, "rendered_diffuse")
    local_feature_path = os.path.join(model_path, f"{name}_{id}", output_tag, "rendered_local_feature")
    local_usage_path = os.path.join(model_path, f"{name}_{id}", output_tag, "rendered_local_usage")
    # depth_guidance_checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
    # pipe = pp("depth-estimation", model=depth_guidance_checkpoint, device="cuda")
    # pipe.model.eval()

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(predicted_depth_path, exist_ok=True)
    makedirs(depth_normal_path, exist_ok=True)
    makedirs(rendered_normal_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(feature_path, exist_ok=True)
    makedirs(spec_path, exist_ok=True)
    makedirs(alpha_path, exist_ok=True)
    makedirs(in_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(delta_normal_path, exist_ok=True)
    makedirs(diffuse_path, exist_ok=True)
    makedirs(local_feature_path, exist_ok=True)
    makedirs(local_usage_path, exist_ok=True)
    timestamp_first = 0
    # cnts = []
    # roots = []
    # gaussians_segments = []

    # roots.append(1)
    # cnts.append(tgh.layers[0][0]._xyz.shape[0])
    # gaussians_segments.append(tgh.layers[0][0])
    # current_length = 10
    # for level in range(1, 10):
    #     #segment_count = (math.ceil((10) / current_length)) + 1
    #     # gaussians_segments.clear()
    #     # _scaling = torch.empty(0)
    #     # _rotation = torch.empty(0)
    #     # _velocity = torch.empty(0)
    #     segment_count = len(tgh.layers[level])
    #     #roots.append(segment_count)
    #     act_seg_cnt = 0
    #     for ind in range(segment_count):
    #         # if (current_length * ind - (current_length /4)) > 2:
    #         #     break
    #         #point_cnt += tgh.layers[level][ind]._xyz.shape[0]
    #         cnts.append(tgh.layers[level][ind]._xyz.shape[0])
    #         gaussians_segments.append(tgh.layers[level][ind])
    #         act_seg_cnt += 1
    #     #print("act_seg_cnt", act_seg_cnt)
    #     #print(len(gaussians_segments))
    #     roots.append(act_seg_cnt)
    #     current_length /= 2
    # print("segment counts", len(gaussians_segments))
    # gaussians.clone_from_cpu(gaussians_segments)
    # print("roots", roots)
    # #print("point count", point_cnt)
    # gaussians.save_ply_w_cnts2("./", cnts, roots)
    if len(views) == 0:
        print("No views to render.")
        return
    manifest_rows = []
    psnr_avg = 0.0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        #rendering = render(view[1].cuda(), gaussians, pipeline, background)["render"]
        viewpoint_cam = view[2].cuda()
        # if not (viewpoint_cam.image_name.endswith("000069") or viewpoint_cam.image_name.endswith("000071") or viewpoint_cam.image_name.endswith("000070")):
        #     continue
        #viewpoint_cam.timestamp += 1/60
        timestamp = viewpoint_cam.timestamp
        output_stem = '{0:05d}'.format(idx)
        manifest_rows.append(f"{output_stem}.png\t{viewpoint_cam.image_name}\t{float(timestamp):.10g}\n")
        # if timestamp_first < 0:
        #     timestamp_first +=1
        # if timestamp_first == 0:
        #     timestamp_first = timestamp
        #print(timestamp)
        #print(timestamp_first)
        print(viewpoint_cam.image_height, viewpoint_cam.image_width)
        if use_tgh:
            tgh.put_current_related_gaussians(timestamp, gaussians, True)
        else:
            gaussians.set_current_timestamp(timestamp)
        velocity2 = gaussians.get_velocity2
        print_tensor_distribution("velocity2[..., 0]", velocity2[..., 0:1])
        #gaussians._feature_rest = None
        #timestamp = viewpoint_cam.timestamp
        time_range = viewpoint_cam.timestamp - gaussians.get_t
        #time_range_offset = torch.abs(time_range) + 0.5
        # time_range2 = time_range_offset * time_range_offset
        # time_range3 = time_range_offset**2 * time_range
        # time_range2 = time_range**2
        # time_range3 = time_range**3
        xyz = gaussians.get_xyz + gaussians.get_velocity * time_range / gaussians.get_sigma_t_fixed
        #xyz = gaussians.get_xyz + (gaussians.get_velocity * time_range + gaussians.get_velocity2 * time_range2 + gaussians.get_velocity3 * time_range3) / (gaussians.get_sigma_t + 1)
        #xyz = gaussians.get_xyz + gaussians.get_velocity * time_range# + gaussians.get_velocity2 * time_range2 + gaussians.get_velocity3 * time_range3
        #rot = gaussians.get_rotation + gaussians.get_rot_velocity * (viewpoint_cam.timestamp - gaussians.get_t)
        mt = gaussians.get_temporal_opacity_factor(timestamp=viewpoint_cam.timestamp)
        opacity = gaussians.get_opacity * mt
        #opacity = torch.sigmoid((opacity - 0.5) * 14)
        shs = gaussians.get_features
        render_iteration = iteration
        #shs = None
        # ma = torch.ones(opacity.shape[0], dtype=torch.bool, device=opacity.device)
        # if iteration <= 5000:
        #     pass
        # elif iteration <= 10000:
        #     ma = torch.rand(gaussians.get_opacity.shape[0], device=gaussians.get_opacity.device).cuda()
        #     ma = ma < (1 - 0.1)
        # elif iteration <= 30000:
        #     ma = torch.rand(gaussians.get_opacity.shape[0], device=gaussians.get_opacity.device).cuda()
        #     ma = ma < (1 - 0.2)
        # else:
        #     ma = torch.rand(gaussians.get_opacity.shape[0], device=gaussians.get_opacity.device).cuda()
        #     ma = ma < (1 - 0.3)
        # ma = torch.logical_and(ma, (mt > 0.05).squeeze())
        ma = (mt > 0.05).squeeze()
        local_mt = local_gaussians.get_temporal_opacity_factor(viewpoint_cam.timestamp)
        local_ma = (local_mt > 0.05).squeeze()
        print("active sh", gaussians.active_sh_degree)
        gaussians.brdf_mlp.build_mips()
        view_pos = viewpoint_cam.camera_center.repeat(gaussians.get_opacity.shape[0], 1) 
        d_viewdir_normalized = safe_normalize(view_pos - xyz)
        normal = gaussians.get_normal(viewpoint_cam.camera_center, xyz)
        normal = normal + gaussians.get_delta_normal
        reflvec = safe_normalize(reflect(d_viewdir_normalized, normal))
        dir_pp = (xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1)).detach()
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        render_package = render_3d_pgsr_anti(viewpoint_cam, xyz, None, opacity, gaussians.active_sh_degree,
                    gaussians.get_scaling, gaussians.get_rotation, background, shs=shs, mask=ma, local_mask=local_ma, max_sh_channels=gaussians.max_sh_degree, normal=normal, reflect=reflvec, dir_pp=dir_pp_normalized, pc=gaussians, local_pc=local_gaussians, iteration=render_iteration, timestamp=timestamp, local_feature_start_iter=int(getattr(pipeline, "local_feature_start_iter", 12000)))
        #rendering = render_3d_pgsr_anti(viewpoint_cam, xyz, None, opacity, gaussians.active_sh_degree, 
        #                           gaussians.get_scaling, gaussians.get_rotation, background, shs=shs, mask=ma, max_sh_channels=gaussians.max_sh_degree)["render"]
        rendering = render_package["render"]
        depth_normal = (render_package["depth_normal"] + 1.0) / 2
        rendered_normal = (render_package["rendered_normal"] + 1.0) / 2
        render_depth = render_package["depth"]
        gt = view[0][0:3, :, :].cuda()
        error_map = torch.abs(rendering - gt)
        #feature_map = render_package["rendered_feature"].detach()
        feature_map = render_package["feature_map"]
        local_feature_map = render_package["local_feature_map"]
        spec_rgb = render_package["spec_rgb"]
        render_alpha = render_package["alpha"]
        render_in = render_package["rendered_in"]
        rendered_delta_normal = render_package["rendered_delta_normal"]
        render_diffuse = render_package["rendered_diffuse"]
        psnr_avg += psnr(rendering.clamp(0.0, 1.0), gt.cuda())
        # h, w = feature_map.shape[1:]
        # flat_feature = feature_map.permute(1, 2, 0).reshape(-1, 4)
        # flat_mean = flat_feature.mean(dim=0, keepdim=True)
        # centered_feature = flat_feature - flat_mean
        # if centered_feature.abs().max() > 0:
        #     _, _, pcs = torch.pca_lowrank(centered_feature, q=3)
        #     projected = centered_feature @ pcs[:, :3]
        #     feature_vis = projected.reshape(h, w, 3).permute(2, 0, 1)
        #     feature_vis = feature_vis - feature_vis.min()
        #     feature_vis = feature_vis / (feature_vis.max() - feature_vis.min() + 1e-6)
        # else:
        #     feature_vis = torch.zeros((3, h, w), device=feature_map.device)

        # to_pil_image = transforms.ToPILImage()
        # gt_pil = to_pil_image(gt)
        # sgt_depth = pipe(gt_pil)
        # predicted_depth = sgt_depth["predicted_depth"]

        print("image size")
        print(depth_normal.size(), gt.size())
        render_output = torch.cat((rendering.clamp(0.0, 1.0), gt.clamp(0.0, 1.0)), dim=2) if append_gt else rendering
        torchvision.utils.save_image(render_output, os.path.join(render_path, output_stem + ".png"))
        if skip_save is not None and skip_save:
            continue
        _usage = render_package["local_light_usage"].detach().float()
        _usage_map = torch.zeros(viewpoint_cam.H * viewpoint_cam.W, 3, device=_usage.device)
        _usage_map[render_package["select_index"]] = _usage
        torchvision.utils.save_image(
            _usage_map.reshape(viewpoint_cam.H, viewpoint_cam.W, 3).permute(2, 0, 1).clamp(0.0, 1.0),
            os.path.join(local_usage_path, output_stem + ".png"))
        #cv2.imwrite("./debug_render_{}.jpg".format(viewpoint_cam.image_name), ((rendering.clip(min=0, max=1).squeeze().permute(1,2,0).detach().cpu().numpy()[..., [2,1,0]] * 255).astype(np.uint8)))
        torchvision.utils.save_image(gt, os.path.join(gts_path, output_stem + ".png"))
        torchvision.utils.save_image(rendered_normal, os.path.join(rendered_normal_path, output_stem + ".png"))
        torchvision.utils.save_image(depth_normal, os.path.join(depth_normal_path, output_stem + ".png"))
        #predicted_depth_image = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
        #torchvision.utils.save_image(predicted_depth_image, os.path.join(predicted_depth_path, '{0:05d}'.format(idx) + ".png"))
        #np.save(os.path.join(predicted_depth_path, '{0:05d}'.format(idx) + ".npy"), predicted_depth)
        render_depth_image = (render_depth - render_depth.min()) / (render_depth.max() - render_depth.min())
        torchvision.utils.save_image(render_depth_image, os.path.join(depth_path, output_stem + ".png"))
        torchvision.utils.save_image((feature_map[0:3] + 1) / 2, os.path.join(feature_path, output_stem + ".png"))
        torchvision.utils.save_image((local_feature_map[0:3] + 1) / 2, os.path.join(local_feature_path, output_stem + ".png"))
        torchvision.utils.save_image((render_alpha - 0.9) * 10, os.path.join(alpha_path, output_stem + ".png"))
        torchvision.utils.save_image(render_in, os.path.join(in_path, output_stem + ".png"))
        torchvision.utils.save_image(error_map, os.path.join(error_path, output_stem + ".png"))
        if spec_rgb is not None:
            torchvision.utils.save_image(spec_rgb, os.path.join(spec_path, 'spec_rgb_' + output_stem + ".png"))
        if rendered_delta_normal is not None:
            torchvision.utils.save_image((rendered_delta_normal + 1) / 2, os.path.join(delta_normal_path, output_stem + ".png"))
        if render_diffuse is not None:
            torchvision.utils.save_image(render_diffuse, os.path.join(diffuse_path, output_stem + ".png"))
    with open(os.path.join(model_path, f"{name}_{id}", output_tag, "manifest.tsv"), "w") as manifest_file:
        manifest_file.write("file\timage_name\ttimestamp\n")
        manifest_file.writelines(manifest_rows)
    psnr_avg /= len(views)
    #print(psnr_avg.shape)
    print("Average PSNR: {:.2f}".format(psnr_avg.mean()))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_save : bool, id : int, render_suffix="", render_frames="", zero_local_gaussians=False, append_gt=False):
    with torch.no_grad():
        tgh = TemperalGaussianHierarchy(dataset.sh_degree, 9, 10,  gaussian_dim=4, time_duration=[0, 30], rot_4d=True, force_sh_3d=False, sh_degree_t=2)
        gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=4, rot_4d=True)
        local_gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=4, rot_4d=True)
        _env_center = torch.tensor(
            [float(v) for v in str(getattr(pipeline, "env_sphere_center", "0,0,0")).split(",")],
            device="cuda", dtype=torch.float32)
        for _m in (gaussians, local_gaussians):
            _m.env_center = _env_center
            _m.env_radius = float(getattr(pipeline, "env_sphere_radius", 8.0))
            _m.inside_diffuse_source = str(getattr(pipeline, "inside_diffuse_source", "albedo"))
            _m.spec_light_combine = str(getattr(pipeline, "spec_light_combine", "exp_sum"))
            _m.local_light_mlp_geo_inputs = bool(getattr(pipeline, "local_light_mlp_geo_inputs", True))
            _m.local_light_mlp_detach_cos = bool(getattr(pipeline, "local_light_mlp_detach_cos", False))
        gaussians.temporal_opacity_mode = pipeline.temporal_opacity_mode
        gaussians.temporal_flat_radius_mult = pipeline.temporal_flat_radius_mult
        gaussians.temporal_flat_edge_sigma_mult = pipeline.temporal_flat_edge_sigma_mult
        local_gaussians.temporal_opacity_mode = pipeline.temporal_opacity_mode
        _local_topa_override = str(getattr(pipeline, "local_temporal_opacity_mode", "") or "")
        local_gaussians.temporal_opacity_mode_override = _local_topa_override
        if _local_topa_override:
            local_gaussians.temporal_opacity_mode = _local_topa_override
        local_gaussians.temporal_flat_radius_mult = pipeline.temporal_flat_radius_mult
        local_gaussians.temporal_flat_edge_sigma_mult = pipeline.temporal_flat_edge_sigma_mult
        gaussians.fourier_c2f_start_iter = getattr(pipeline, "fourier_c2f_start_iter", 30_000)
        gaussians.fourier_c2f_end_iter = getattr(pipeline, "fourier_c2f_end_iter", -1)
        print("Temporal opacity mode:", pipeline.temporal_opacity_mode)
        print("Temporal flat legacy args:", pipeline.temporal_flat_radius_mult, pipeline.temporal_flat_edge_sigma_mult)
        scene = Scene(dataset, gaussians, tgh, local_gaussians=local_gaussians, shuffle=False, render_only=True, eid=id)
        if zero_local_gaussians:
            initialize_local_gaussian_model(gaussians, local_gaussians, zero_local_feature=True)
            print("Forcing zero-initialized local Gaussians for rendering.")
        elif not getattr(scene, "local_gaussians_loaded", False):
            initialize_local_gaussian_model(gaussians, local_gaussians, zero_local_feature=True)
            print("No saved local Gaussian state; using zero local feature fallback.")
        else:
            print("Using restored local Gaussian checkpoint.")
        current_length = tgh.max_layer_length
        point_cnt = 0
        # for level in range(0, 10):
        #     #segment_count = (math.ceil((10) / current_length)) + 1
        #     segment_count = len(tgh.layers[level])
        #     for ind in range(segment_count):
        #         point_cnt += tgh.layers[level][ind]._xyz.shape[0]
        #     current_length /= 2
        # print("point count", point_cnt)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        use_tgh = getattr(scene, "loaded_tgh_checkpoint", False)

        frame_filter = _parse_render_frames(render_frames)

        if not skip_train:
               train_views = _filter_views_by_frames(scene.getTrainCameras(), frame_filter, "train")
               render_set(dataset.model_path, "train", scene.loaded_iter, train_views, gaussians, local_gaussians, tgh, pipeline, background, skip_save, id, use_tgh=use_tgh, render_suffix=render_suffix, append_gt=append_gt)

        if not skip_test:
               test_views = _filter_views_by_frames(scene.getTestCameras(), frame_filter, "test")
               render_set(dataset.model_path, "test", scene.loaded_iter, test_views, gaussians, local_gaussians, tgh, pipeline, background, skip_save, id, use_tgh=use_tgh, render_suffix=render_suffix, append_gt=append_gt)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--config", default="", type=str,
                        help="Training yaml or effective_config_*.yaml; its values override cfg_args/defaults, explicit CLI flags override it.")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_save", action="store_true")
    parser.add_argument("--render_suffix", default="", type=str)
    parser.add_argument("--render_frames", default="", type=str, help="Comma-separated frame ids or ranges, e.g. 69,70,71 or 69-71")
    parser.add_argument("--zero_local_gaussians", action="store_true", help="Ignore any saved local checkpoint and render with zero-initialized local Gaussians.")
    parser.add_argument("--append_gt", action="store_true", help="Append the ground-truth image to the right of each rendered image.")
    args = get_combined_args(parser)
    args = _apply_render_config(args, parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    id = _infer_checkpoint_id(args.loaded_pth)
    print("Rendering id:", id)
    pipeline_args = pipeline.extract(args)
    # train-only keys render still consumes (not part of PipelineParams)
    if "local_feature_start_iter" in args.extra_cfg:
        pipeline_args.local_feature_start_iter = int(args.extra_cfg["local_feature_start_iter"])
    render_sets(model.extract(args), args.iteration, pipeline_args, args.skip_train, args.skip_test, args.skip_save, id, args.render_suffix, args.render_frames, args.zero_local_gaussians, args.append_gt)
