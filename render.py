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
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, tgh, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    timestamp_first = 0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        #rendering = render(view[1].cuda(), gaussians, pipeline, background)["render"]
        viewpoint_cam = view[2].cuda()
        #viewpoint_cam.timestamp += 1/60
        timestamp = viewpoint_cam.timestamp
        # if timestamp_first < 0:
        #     timestamp_first +=1
        # if timestamp_first == 0:
        #     timestamp_first = timestamp
        print(timestamp)
        print(timestamp_first)
        print(viewpoint_cam.image_height, viewpoint_cam.image_width)
        tgh.put_current_related_gaussians(timestamp, gaussians, True)
        #timestamp = viewpoint_cam.timestamp
        xyz = gaussians.get_xyz + gaussians.get_velocity * (viewpoint_cam.timestamp - gaussians.get_t) / (gaussians.get_sigma_t + 1)
        mt = gaussians.get_marginal_t(timestamp=viewpoint_cam.timestamp)
        opacity = gaussians.get_opacity * mt
        shs = gaussians.get_features
        ma = (mt > 0.05).squeeze()
        print("active sh", gaussians.active_sh_degree)
        render_package = render_3d_pgsr_anti(viewpoint_cam, xyz, None, opacity, gaussians.active_sh_degree,
                                                    gaussians.get_scaling, gaussians.get_rotation, background, shs=shs, mask=ma, max_sh_channels=gaussians.max_sh_degree)
        #rendering = render_3d_pgsr_anti(viewpoint_cam, xyz, None, opacity, gaussians.active_sh_degree, 
        #                           gaussians.get_scaling, gaussians.get_rotation, background, shs=shs, mask=ma, max_sh_channels=gaussians.max_sh_degree)["render"]
        rendering = render_package["render"]
        depth_normal = render_package["depth_normal"]
        rendered_normal = render_package["rendered_normal"]
        gt = view[0][0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        tgh = TemperalGaussianHierarchy(dataset.sh_degree, 9, 10,  gaussian_dim=4, time_duration=[0, 10], rot_4d=True, force_sh_3d=False, sh_degree_t=2)
        gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=4, rot_4d=True)
        scene = Scene(dataset, gaussians, tgh, shuffle=False, render_only=True)

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
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)