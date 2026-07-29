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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for key, default in vars(self).items():
            name = key[1:] if key.startswith("_") else key
            setattr(group, name, getattr(args, name, default))
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.extension = ".png"
        self.num_extra_pts = 0
        self.loaded_pth = ""
        self.frame_ratio = 1
        self.frame_filter = ""
        self.dataloader = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.env_map_res = 0
        self.env_optimize_until = 1000000000
        self.env_optimize_from = 0
        self.eval_shfs_4d = False
        self.temporal_opacity_mode = "normalized_sigmoid"
        # local-branch temporal opacity mode override: "" = same as global
        # (temporal_opacity_mode); "gaussian" = plain marginal for locals (the
        # local effect-range penalties are then disabled and local init widens
        # scaling_t per local_temporal_init_frames)
        self.local_temporal_opacity_mode = ""
        self.temporal_flat_radius_mult = 0.75
        self.temporal_flat_edge_sigma_mult = 2.0
        self.sph_residual_keyframes = 0
        self.sph_residual_from_iter = 20_000
        self.sph_hierarchy_bands = ""
        self.sph_parity_bands = ""
        self.sph_sliding_bands = ""
        self.sph_sliding_window = 16
        self.sph_time_min = -1.0
        self.sph_time_max = -1.0
        # Coarse-to-fine temporal Fourier features (specular2 coeffs): bands
        # open low-frequency-first between start and end iter (BARF-style
        # smooth window). end <= start (default) = original hard gate at start.
        self.fourier_c2f_start_iter = 30_000
        self.fourier_c2f_end_iter = -1
        # environment sphere separating scene content from env/background points
        # (base-color source, densify inside/outside gating).  Defaults = the
        # historical hardcoded abuzabi object values.
        self.env_sphere_center = "0,0,0"
        self.env_sphere_radius = 8.0
        # diffuse color for inside-sphere points after lighting_start_iter:
        # "albedo" (historical, learned albedo initialized from SH) or "sh_dc"
        # (0-degree SH color evaluated directly, view-independent)
        self.inside_diffuse_source = "albedo"
        # zero-init light_mlp_2's final-layer weights so the local-feature
        # pathway starts with zero contribution to spec_light (neutral exp
        # factor) and only grows as needed - keeps diffuse brightness on the
        # base color instead of being absorbed via local features
        self.local_light_mlp_zero_init = False
        # how the two light MLPs combine into spec_light:
        #   "exp_sum" (historical): exp(mlp_base + mlp2) - multiplicative interaction
        #   "sum_exp": exp(mlp_base) + exp(mlp2) - two additive positive light
        #     terms; with local_light_mlp_zero_init the mlp2 term starts at a
        #     TRUE zero (~exp(-5)) instead of a neutral factor
        self.spec_light_combine = "exp_sum"
        # include roughness + cos(normal, reflection) in light_mlp_2's input
        # (historical True). False removes them, severing the gradient path from
        # the local light branch into geometry/normals.
        self.local_light_mlp_geo_inputs = True
        # keep cos_nr as an mlp2 input but stop its gradient into the normals
        self.local_light_mlp_detach_cos = False
        # append the Schlick Fresnel basis (1 - clamp(cos_nr, 0, 1))^5 as one
        # extra input column to the GLOBAL light_mlp so the env light can learn
        # grazing-angle (Fresnel) falloff.  Widens light_mlp input by 1;
        # resuming an older checkpoint zero-init widens the first layer via
        # ensure_parity_light_env (function-preserving, influence grows from 0).
        self.light_mlp_fresnel_input = False
        # append raw cos_nr (= cos(normal, reflection) = n.v, clamped to
        # [-1, 1] at computation) as one extra GLOBAL light_mlp input column:
        # smooth full-range angular dependence (Ref-NeRF style), more flexible
        # but more shortcut-prone than the Schlick basis.  Same widening /
        # resume behavior; column appended after the fresnel one.
        self.light_mlp_cos_input = False
        # detach cos_nr inside the GLOBAL light_mlp fresnel/cos input columns
        # (mirrors local_light_mlp_detach_cos): the angular value still informs
        # the MLP in the forward pass, but photometric gradients can no longer
        # bend the normals through these columns — d(1-cos)^5 = -5(1-cos)^4
        # blows up exactly at grazing angles where the floor reflections live,
        # the suspected instability of the non-detached columns.  Normals keep
        # their usual gradient paths (reflection-dir env sampling, geometry
        # losses).  Function-identical forward, so it can be toggled on resume.
        self.light_mlp_detach_cos = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_t_lr_init = -1.0
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.brdf_mlp_lr_init = 0.016
        self.brdf_mlp_lr_final = 0.00016
        self.brdf_mlp_lr_delay_mult = 0.01
        self.brdf_mlp_lr_max_steps = 30_000
        self.encoding_lr_init = 0.002
        self.encoding_lr_final = 0.001
        self.encoding_lr_delay_mult = 0.1
        self.encoding_lr_max_steps = 30_000
        self.mlp_lr_init = 0.0005
        self.mlp_lr_final = 0.0002
        self.mlp_lr_delay_mult = 0.1
        self.mlp_lr_max_steps = 30_000
        self.feature_lr = 0.002
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        # optional rotation LR decay (constant-LR groups carry tail noise);
        # <= 0 = historical constant behavior
        self.rotation_lr_final = -1.0
        self.rotation_lr_delay_mult = 1.0
        self.rotation_lr_max_steps = -1
        self.velocity2_lr_init = -1.0
        self.velocity2_lr_final = -1.0
        self.velocity2_lr_delay_mult = -1.0
        self.velocity2_lr_max_steps = -1
        self.specular_lr = 0.0001
        self.albedo_lr = 0.001
        self.delta_normal_lr = 0.001
        self.roughness_lr = 0.0002
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.thresh_opa_prune = 0.005
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        # self.densify_from_iter2 = 20_000
        # self.densify_until_iter2 = 25_000
        self.densify_grad_threshold = 0.0002
        self.densify_grad_t_threshold = 0.002
        # temporal split t-grad selection: top (1 - quantile) fraction of points
        # by accumulated |dL/dt| are split each event (0.99 = top 1%, the exp78
        # record setting; lower = more temporal splits, higher = fewer)
        self.densify_grad_t_quantile = 0.98
        # absolute floor on the quantile-derived |dL/dt| threshold: once the
        # population converges below the floor, no more temporal splits happen
        # (0 = pure quantile, the exp78 behavior; measured signal scale in
        # abuzabi is ~5.5-6.3e-5, so ~1e-5 is a gentle floor)
        self.densify_grad_t_floor = 0.0
        # scheduled batch-size switch: at batch_size_final_from_iter the training
        # loader is rebuilt with batch_size_final views per step (-1 = disabled).
        # Gradient-noise reduction for the LR tail; iteration count is unchanged.
        self.batch_size_final = -1
        self.batch_size_final_from_iter = 50_000
        # generalized: "iter:bs,iter:bs,..." switches AT iter (supports down-
        # switching); empty = use batch_size_final; overrides it when set.
        self.batch_size_schedule = ""
        # Training checkpoints are written before the optimizer step bearing the
        # checkpoint's iteration label.  Enable this for controlled branches that
        # must replay that pending iteration rather than start at label + 1.
        self.resume_replay_checkpoint_iteration = False
        # phase-start iterations (previously hardcoded), exposed so schedules can
        # be rescaled e.g. for larger batch sizes. Defaults = historical values.
        self.lighting_start_iter = 9000
        self.local_feature_start_iter = 12000
        self.reset_opacity_high_until_iter = 25_000
        # densification interval staircase: "until:interval,..." (iteration <
        # until; -1 = catch-all). Default = historical hardcoded schedule.
        self.densify_interval_schedule = "3001:100,15000:200,25000:300,35000:500,45000:500,-1:1000"
        # per-frame point-cloud init window (inclusive frame indices into
        # pcds_j10).  -1 = historical hardcoded behavior: the point-budget
        # pre-pass counts frames 19..80 while the load loop takes 0..100
        # bounded by time_duration.  Setting both applies the same window to
        # count and load, sizing the per-frame point budget consistently.
        self.pcd_init_frame_start = -1
        self.pcd_init_frame_end = -1
        # scene time model (previously hardcoded for 60 frames @30fps):
        # frame-count-based window losses use temporal_fps; the clip t-range
        # clamps anchor temporal centers inside [t_min, t_max].
        self.temporal_fps = 30.0
        self.temporal_cap_frames = 32.0
        self.temporal_min_effect_frames = 0.5
        self.temporal_clip_t_min = 0.6666666666666666
        self.temporal_clip_t_max = 2.6333333333333333
        # opacity-loss schedules (previously hardcoded; defaults = historical).
        # Global sparsity loss also requires densify_from < iter <= densify_until
        # (unchanged); from_iter -1 = no extra lower bound, until -1 = no end.
        self.global_opacity_loss_from_iter = -1
        self.global_opacity_loss_until_iter = 30_000
        self.global_opacity_loss_weight = 0.02
        self.local_opacity_loss_from_iter = 15_000
        self.local_opacity_loss_until_iter = -1
        self.local_opacity_loss_weight = 0.01
        # local light_mlp_2 output penalty ("pay for local light usage"): acts on
        # the branch output that reaches the image instead of point opacities.
        # sum_exp: mean additive local radiance; exp_sum: mean |mlp_output|.
        # 0 = off (historical behavior).
        self.local_mlp_output_loss_weight = 0.0
        self.local_mlp_output_loss_from_iter = 0
        self.local_mlp_output_loss_until_iter = -1
        # local feature decay: L1 pull of the visible locals' carried feature
        # (get_specular) toward zero; the normalized local feature map returns
        # to gray wherever the reflection gradient stops defending the feature.
        # 0 = off (historical behavior).
        self.local_feature_loss_weight = 0.0
        self.local_feature_loss_from_iter = 0
        self.local_feature_loss_until_iter = -1
        # local alpha completeness: pull the local pass's rendered alpha toward
        # 1 at the inside-sphere shading pixels (partial coverage = mixed /
        # unstable normalized feature directions). 0 = off.
        self.local_alpha_loss_weight = 0.0
        self.local_alpha_loss_from_iter = 0
        self.local_alpha_loss_until_iter = -1
        # gaussian-mode local init: <= 0 (default) = per-gaussian match to the
        # flat-window >0.05 effect range at clone time (statics stay wide,
        # dynamics stay narrow); > 0 = uniform range of N frames (full width)
        self.local_temporal_init_frames = -1.0
        # local spawn semantics (sc-project match): born feature-black + dim so
        # the local feature map starts gray and only develops where reflection
        # gradients paint it.  zero_feature False = clone trained _specular;
        # spawn_opacity outside (0,1) = keep cloned opacities.
        self.local_spawn_zero_feature = True
        self.local_spawn_opacity = 0.1
        # local prune: lower opacity bar than the global thresh_opa_prune, and
        # only a random fraction of eligible candidates removed per prune event
        # (gradual drain keeps the normalized local feature map stable;
        # 1.0 = prune all candidates, <= 0 = never opacity/size-prune locals)
        self.local_thresh_opa_prune = 0.01
        self.local_prune_sample_fraction = 0.1
        # local reset_opacity_high (>0.995 -> 0.99) runs on the opacity reset
        # cadence while iteration <= this; negative disables
        self.local_reset_opacity_high_until_iter = 30_000
        self.density_entropy_loss_from_iter = 3_000
        self.density_entropy_loss_weight = 0.01
        # monocular prior losses (sgt_depth / sgt_normal) stop iterations;
        # -1 = never stop. Defaults = historical hardcoded schedule.
        self.mono_depth_loss_until_iter = 40_000
        self.mono_normal_loss_until_iter = 20_000
        # LPIPS loss applies while iteration <= this; -1 = never stop
        self.lpips_loss_until_iter = 5_000
        self.densify_specular_time_threshold = 0.0000003
        self.temporal_split_from_iter = 25_000
        self.temporal_split_until_iter = 35_000
        self.opacity_zero_reset_iter = -1
        self.opacity_zero_reset_value = 1.0e-6
        self.opacity_periodic_reset_interval = -1
        self.opacity_periodic_reset_value = 0.1
        self.opacity_periodic_reset_from_iter = 20_000
        self.opacity_periodic_reset_until_iter = -1
        self.temporal_opacity_k_start = 2.0
        self.temporal_opacity_k_final = 12.0
        self.temporal_opacity_k_ramp_start = 25_000
        self.temporal_opacity_k_ramp_end = 45_000
        self.temporal_opacity_k_loss_weight = 0.01
        self.temporal_flat_range_level = 0.05
        # Legacy config keys kept for compatibility; flat_window now uses a learned per-Gaussian radius.
        self.temporal_flat_radius_start_mult = 0.0
        self.temporal_flat_radius_final_mult = 0.75
        self.temporal_flat_radius_ramp_start = 25_000
        self.temporal_flat_radius_ramp_end = 35_000
        self.temporal_flat_radius_loss_weight = 0.0
        self.train_num_workers = -1
        self.debug_interval = 50
        self.local_branch_lazy_init = False
        self.densify_until_num_points = -1
        self.final_prune_from_iter = -1
        self.sh_increase_interval = 1000
        self.lambda_opa_mask = 0.0
        self.lambda_rigid = 0.0
        self.lambda_motion = 0.0
        # Per-camera affine color compensation (PGSR-style): exp(a)*render + b
        # applied to the L1 term only for TRAIN cameras; test views always render
        # uncompensated.  Off by default.
        self.cam_affine_enable = False
        self.cam_affine_lr = 0.001
        self.cam_affine_from_iter = 1000
        self.cam_affine_ssim_gate = 0.5

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except (TypeError, FileNotFoundError):
        print("Config file not found, skipping cfg_args")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
