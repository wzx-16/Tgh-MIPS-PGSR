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
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import math
from torchvision import transforms
from transformers import pipeline as pp
import numpy as np
import cv2

def render_set(model_path, name, iteration, views, gaussians, tgh, pipeline, background, id):
    render_path = os.path.join(model_path, f"{name}_{id}", "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, f"{name}_{id}", "ours_{}".format(iteration), "gt")
    predicted_depth_path = os.path.join(model_path, f"{name}_{id}", "ours_{}".format(iteration), "predicted_depth")
    depth_normal_path = os.path.join(model_path, f"{name}_{id}", "ours_{}".format(iteration), "depth_normal")
    rendered_normal_path = os.path.join(model_path, f"{name}_{id}", "ours_{}".format(iteration), "rendered_normal")
    depth_path = os.path.join(model_path, f"{name}_{id}", "ours_{}".format(iteration), "rendered_depth")
    feature_path = os.path.join(model_path, f"{name}_{id}", "ours_{}".format(iteration), "rendered_feature")
    spec_path = os.path.join(model_path, f"{name}_{id}", "ours_{}".format(iteration), "rendered_specular")
    alpha_path = os.path.join(model_path, f"{name}_{id}", "ours_{}".format(iteration), "rendered_alpha")
    in_path = os.path.join(model_path, f"{name}_{id}", "ours_{}".format(iteration), "rendered_in")
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
        xyz = gaussians.get_xyz + gaussians.get_velocity * time_range / (gaussians.get_sigma_t + 1)
        #xyz = gaussians.get_xyz + (gaussians.get_velocity * time_range + gaussians.get_velocity2 * time_range2 + gaussians.get_velocity3 * time_range3) / (gaussians.get_sigma_t + 1)
        #xyz = gaussians.get_xyz + gaussians.get_velocity * time_range# + gaussians.get_velocity2 * time_range2 + gaussians.get_velocity3 * time_range3
        #rot = gaussians.get_rotation + gaussians.get_rot_velocity * (viewpoint_cam.timestamp - gaussians.get_t)
        mt = gaussians.get_marginal_t(timestamp=viewpoint_cam.timestamp)
        #mt = torch.sigmoid((mt - 0.5) * 14)
        #mt = torch.sigmoid((mt - 0.5) * (12 + gaussians.get_specular[..., 0:1]))
        scaler = (torch.sigmoid(0.5 * (gaussians.get_specular[..., 0:1])) - torch.sigmoid(-0.5 * (gaussians.get_specular[..., 0:1])))
        min_opa = torch.sigmoid(-0.5 * (gaussians.get_specular[..., 0:1]))
        mt = (torch.sigmoid((mt - 0.5) * (gaussians.get_specular[..., 0:1])) - min_opa) / scaler
        opacity = gaussians.get_opacity * mt
        #opacity = torch.sigmoid((opacity - 0.5) * 14)
        shs = gaussians.get_features
        iteration = 60000
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
                                                    gaussians.get_scaling, gaussians.get_rotation, background, shs=shs, mask=ma, max_sh_channels=gaussians.max_sh_degree, normal=normal, reflect=reflvec, dir_pp=dir_pp_normalized, pc=gaussians, iteration=iteration, timestamp=timestamp)
        #rendering = render_3d_pgsr_anti(viewpoint_cam, xyz, None, opacity, gaussians.active_sh_degree, 
        #                           gaussians.get_scaling, gaussians.get_rotation, background, shs=shs, mask=ma, max_sh_channels=gaussians.max_sh_degree)["render"]
        rendering = render_package["render"]
        depth_normal = (render_package["depth_normal"] + 1.0) / 2
        rendered_normal = (render_package["rendered_normal"] + 1.0) / 2
        render_depth = render_package["depth"]
        gt = view[0][0:3, :, :]
        #feature_map = render_package["rendered_feature"].detach()
        feature_map = render_package["feature_map"]
        spec_rgb = render_package["spec_rgb"]
        render_alpha = render_package["alpha"]
        render_in = render_package["rendered_in"]
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
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        #cv2.imwrite("./debug_render_{}.jpg".format(viewpoint_cam.image_name), ((rendering.clip(min=0, max=1).squeeze().permute(1,2,0).detach().cpu().numpy()[..., [2,1,0]] * 255).astype(np.uint8)))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendered_normal, os.path.join(rendered_normal_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth_normal, os.path.join(depth_normal_path, '{0:05d}'.format(idx) + ".png"))
        #predicted_depth_image = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
        #torchvision.utils.save_image(predicted_depth_image, os.path.join(predicted_depth_path, '{0:05d}'.format(idx) + ".png"))
        #np.save(os.path.join(predicted_depth_path, '{0:05d}'.format(idx) + ".npy"), predicted_depth)
        render_depth_image = (render_depth - render_depth.min()) / (render_depth.max() - render_depth.min())
        torchvision.utils.save_image(render_depth_image, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image((feature_map[0:3] + 1) / 2, os.path.join(feature_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image((render_alpha - 0.9) * 10, os.path.join(alpha_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(render_in, os.path.join(in_path, '{0:05d}'.format(idx) + ".png"))
        if spec_rgb is not None:
            torchvision.utils.save_image(spec_rgb, os.path.join(spec_path, 'spec_rgb_{0:05d}'.format(idx) + ".png"))
    psnr_avg /= len(views)
    #print(psnr_avg.shape)
    print("Average PSNR: {:.2f}".format(psnr_avg.mean()))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, id : int):
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
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, tgh, pipeline, background, id)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, tgh, pipeline, background, id)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    id = args.loaded_pth.split("tgh")[-1][0] if args.loaded_pth else 0
    print("Rendering id:", id)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, id)