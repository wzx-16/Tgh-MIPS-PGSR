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
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
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
        self.densify_grad_t_quantile = 0.99
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
        # phase-start iterations (previously hardcoded), exposed so schedules can
        # be rescaled e.g. for larger batch sizes. Defaults = historical values.
        self.lighting_start_iter = 9000
        self.local_feature_start_iter = 12000
        self.reset_opacity_high_until_iter = 25_000
        # densification interval staircase: "until:interval,..." (iteration <
        # until; -1 = catch-all). Default = historical hardcoded schedule.
        self.densify_interval_schedule = "3001:100,15000:200,25000:300,35000:500,45000:500,-1:1000"
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
        self.density_entropy_loss_from_iter = 3_000
        self.density_entropy_loss_weight = 0.01
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
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
