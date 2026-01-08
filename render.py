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
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_3d_pgsr_anti
import torchvision
from utils.general_utils import safe_state, safe_normalize, reflect
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import math
from torchvision import transforms
from transformers import pipeline as pp
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import sys

def mse(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def get_mae(gt_normal_stack: np.ndarray, render_normal_stack: np.ndarray) -> float:
    MAE = np.mean(np.arccos(np.clip(np.sum(gt_normal_stack * render_normal_stack, axis=-1), -1, 1)) * 180 / np.pi)
    return MAE.item()

def render_set(model_path, name, iteration, views, gaussians, tgh, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    predicted_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "predicted_depth")
    depth_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_normal")
    rendered_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rendered_normal")
    rendered_global_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rendered_global_normal")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rendered_depth")
    feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rendered_feature")
    feature2_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rendered_feature2")
    diffuse_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rendered_diffuse")
    # depth_guidance_checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
    # pipe = pp("depth-estimation", model=depth_guidance_checkpoint, device="cuda")
    # pipe.model.eval()

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(predicted_depth_path, exist_ok=True)
    makedirs(depth_normal_path, exist_ok=True)
    makedirs(rendered_normal_path, exist_ok=True)
    makedirs(rendered_global_normal_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(feature_path, exist_ok=True)
    makedirs(feature2_path, exist_ok=True)
    makedirs(diffuse_path, exist_ok=True)
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
    psnr_avg = 0.0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        #rendering = render(view[1].cuda(), gaussians, pipeline, background)["render"]
        viewpoint_cam = view[2].cuda()
        #viewpoint_cam.timestamp += 1/60
        timestamp = viewpoint_cam.timestamp
        # if timestamp_first < 0:
        #     timestamp_first +=1
        # if timestamp_first == 0:
        #     timestamp_first = timestamp
        #print(timestamp)
        #print(timestamp_first)
        print(viewpoint_cam.image_height, viewpoint_cam.image_width)
        tgh.put_current_related_gaussians(timestamp, gaussians, True)
        #gaussians._feature_rest = None
        #timestamp = viewpoint_cam.timestamp
        time_range = viewpoint_cam.timestamp - gaussians.get_t
        #time_range_offset = torch.abs(time_range) + 0.5
        # time_range2 = time_range_offset * time_range_offset
        # time_range3 = time_range_offset**2 * time_range
        # time_range2 = time_range**2
        # time_range3 = time_range**3
        xyz = gaussians.get_xyz# + gaussians.get_velocity * time_range / (gaussians.get_sigma_t + 1)
        #xyz = gaussians.get_xyz + (gaussians.get_velocity * time_range + gaussians.get_velocity2 * time_range2 + gaussians.get_velocity3 * time_range3) / (gaussians.get_sigma_t + 1)
        #xyz = gaussians.get_xyz + gaussians.get_velocity * time_range# + gaussians.get_velocity2 * time_range2 + gaussians.get_velocity3 * time_range3
        #rot = gaussians.get_rotation + gaussians.get_rot_velocity * (viewpoint_cam.timestamp - gaussians.get_t)
        #mt = gaussians.get_marginal_t(timestamp=viewpoint_cam.timestamp)
        opacity = gaussians.get_opacity# * mt
        #opacity = opacity * 0.7
        shs = gaussians.get_features
        #shs = None
        #ma = (mt > 0.05).squeeze()
        ma = torch.ones(opacity.shape[0], dtype=torch.bool, device=opacity.device)
        # ma = torch.rand(gaussians.get_opacity.shape[0], device=gaussians.get_opacity.device).cuda()
        # ma = ma < (1 - 0.3)
        print("active sh", gaussians.active_sh_degree)
        #gaussians.brdf_mlp.build_mips()
        view_pos = viewpoint_cam.camera_center.repeat(gaussians.get_opacity.shape[0], 1) 
        d_viewdir_normalized = safe_normalize(view_pos - xyz)
        normal = gaussians.get_normal(viewpoint_cam.camera_center, xyz)
        normal = normal + gaussians.get_delta_normal
        reflvec = safe_normalize(reflect(d_viewdir_normalized, normal))
        dir_pp = (xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1)).detach()
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        iteration = int(args.loaded_pth.split("chkpnt")[-1].split(".pth")[0]) if args.loaded_pth else 0
        ENV_CENTER = torch.tensor([-0.2270, 1.9700, 1.7740], device='cuda')
        ENV_RADIUS = 0.974
        render_package = render_3d_pgsr_anti(viewpoint_cam, xyz, None, opacity, gaussians.active_sh_degree,
                                                    gaussians.get_scaling, gaussians.get_rotation, background, shs=shs, mask=ma, max_sh_channels=gaussians.max_sh_degree, normal=normal, delta_normal=gaussians.get_delta_normal[ma], reflect=reflvec, dir_pp=dir_pp_normalized, pc=gaussians, iteration=iteration, ENV_CENTER=ENV_CENTER, ENV_RADIUS=ENV_RADIUS)
        #rendering = render_3d_pgsr_anti(viewpoint_cam, xyz, None, opacity, gaussians.active_sh_degree, 
        #                           gaussians.get_scaling, gaussians.get_rotation, background, shs=shs, mask=ma, max_sh_channels=gaussians.max_sh_degree)["render"]
        rendering = render_package["render"]
        depth_normal = (render_package["depth_normal"] + 1.0) / 2
        rendered_normal = (render_package["rendered_normal"] + 1.0) / 2
        rendered_global_normal = ((render_package["rendered_gb_normal"] + 1.0) / 2)
        render_depth = render_package["depth"]
        if view[0] is not None:
            gt = view[0][0:3, :, :]
        else:
            gt = None
        feature_map = render_package["rendered_feature"].detach()[:3, :, :]
        feature_map2 = render_package["rendered_feature2"].detach()[:3, :, :]
        rendered_diffuse = render_package["rendered_diff"].detach()[:3, :, :]
        if gt is not None:
            one_psnr = psnr(gt.cuda(), rendering).mean()
            psnr_avg += one_psnr.item()
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

        #print("image size")
        #print(depth_normal.size(), gt.size())
        torchvision.utils.save_image(rendering, os.path.join(render_path, viewpoint_cam.image_name.split("/")[-1] + ".jpg"))
        if gt is not None:
            torchvision.utils.save_image(gt, os.path.join(gts_path, viewpoint_cam.image_name.split("/")[-1] + ".jpg"))
        torchvision.utils.save_image(rendered_normal, os.path.join(rendered_normal_path, viewpoint_cam.image_name.split("/")[-1] + ".jpg"))
        torchvision.utils.save_image(rendered_global_normal, os.path.join(rendered_global_normal_path, viewpoint_cam.image_name.split("/")[-1] + ".jpg"))
        torchvision.utils.save_image(depth_normal, os.path.join(depth_normal_path, viewpoint_cam.image_name.split("/")[-1] + ".jpg"))
        #predicted_depth_image = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
        #torchvision.utils.save_image(predicted_depth_image, os.path.join(predicted_depth_path, '{0:05d}'.format(idx) + ".png"))
        #np.save(os.path.join(predicted_depth_path, '{0:05d}'.format(idx) + ".npy"), predicted_depth)
        render_depth_image = (render_depth - render_depth.min()) / (render_depth.max() - render_depth.min())
        torchvision.utils.save_image(render_depth_image, os.path.join(depth_path, viewpoint_cam.image_name.split("/")[-1] + ".jpg"))
        torchvision.utils.save_image((feature_map + 1) / 2, os.path.join(feature_path, viewpoint_cam.image_name.split("/")[-1] + ".jpg"))
        torchvision.utils.save_image((feature_map2 + 1) / 2, os.path.join(feature2_path, viewpoint_cam.image_name.split("/")[-1] + ".jpg"))
        torchvision.utils.save_image(rendered_diffuse, os.path.join(diffuse_path, viewpoint_cam.image_name.split("/")[-1] + ".jpg"))
    print("Average PSNR: {:.2f} dB".format(psnr_avg / len(views)))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, id : int):
    #print("id", id)
    with torch.no_grad():
        tgh = TemperalGaussianHierarchy(dataset.sh_degree, 9, 10,  gaussian_dim=4, time_duration=[0, 30], rot_4d=True, force_sh_3d=False, sh_degree_t=2)
        gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=4, rot_4d=True)
        scene = Scene(dataset, gaussians, tgh, shuffle=False, render_only=True, eid=id)
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

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, tgh, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, tgh, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--id", type=int, default=0)
    args = get_combined_args(parser)
    # parser.add_argument("--config", type=str)
    # parser.add_argument('--debug_from', type=int, default=-1)
    # parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000])
    # parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--start_checkpoint", type=str, default = None)
    
    # parser.add_argument("--gaussian_dim", type=int, default=3)
    # parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    # parser.add_argument('--num_pts', type=int, default=100_000)
    # parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    # parser.add_argument("--rot_4d", action="store_true")
    # parser.add_argument("--force_sh_3d", action="store_true")
    # parser.add_argument("--batch_size", type=int, default=1)
    # parser.add_argument("--seed", type=int, default=6666)
    # parser.add_argument("--exhaust_test", action="store_true")
    
    # #args = parser.parse_args(sys.argv[1:])
    # #args.save_iterations.append(args.iterations)
        
    # cfg = OmegaConf.load(args.config)
    # def recursive_merge(key, host):
    #     if isinstance(host[key], DictConfig):
    #         for key1 in host[key].keys():
    #             recursive_merge(key1, host[key])
    #     else:
    #         assert hasattr(args, key), key
    #         setattr(args, key, host[key])
    # for k in cfg.keys():
    #     recursive_merge(k, cfg)
    # print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    id = args.loaded_pth.split("tgh")[-1][0] if args.loaded_pth else 0
    print("Rendering id:", id)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, id)