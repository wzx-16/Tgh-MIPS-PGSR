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

import os
import torch
import random
import json
from utils.system_utils import searchForMaxIteration, mkdir_p
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.temperal_gaussian_hierarchy import TemperalGaussianHierarchy
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.data_utils import CameraDataset
from scene.NVDIFFREC import save_env_map, load_env

class Scene:

    gaussians : GaussianModel
    tgh : TemperalGaussianHierarchy

    def __init__(self, args : ModelParams, gaussians : GaussianModel, tgh : TemperalGaussianHierarchy, load_iteration=None, shuffle=True, resolution_scales=[1.0], num_pts=100_000, num_pts_ratio=1.0, time_duration=None, render_only=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.tgh = tgh
        self.white_background = args.white_background

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, num_pts_ratio=num_pts_ratio)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, num_pts=num_pts, time_duration=time_duration, extension=args.extension, num_extra_pts=args.num_extra_pts, frame_ratio=args.frame_ratio, dataloader=args.dataloader, render=render_only)
        elif os.path.exists(os.path.join(args.source_path, "calibration_full.json")):
            print("Found calibration_full.json file, assuming THU data set!")
            scene_info = sceneLoadTypeCallbacks["THU"](args.source_path, args.white_background, num_pts=num_pts, time_duration=time_duration, num_extra_pts=args.num_extra_pts, frame_ratio=args.frame_ratio, dataloader=args.dataloader)
        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            print("Found dataset.json, assuming nerfies dataset")
            scene_info = sceneLoadTypeCallbacks["Nerfies"](args.source_path, True)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            if not render_only:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # if shuffle:
        #     random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        #     random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            
        if args.loaded_pth:
            #self.gaussians.create_from_pth(args.loaded_pth, self.cameras_extent)
            #self.gaussians.restore(model_args=torch.load(args.loaded_pth, map_location="cuda:0", weights_only=False)[0], training_args=None)
            (active_sh_degree, spatial_lr_scale, env_map, active_sh_degree_t) = self.tgh.restore(model_args=torch.load(args.loaded_pth, map_location="cpu", weights_only=False)[0], training_args=None)
            self.gaussians.active_sh_degree = active_sh_degree
            self.gaussians.spatial_lr_scale = spatial_lr_scale
            self.gaussians.rot_4d = self.tgh.rot_4d
            if env_map is not None:
                self.gaussians.env_map = env_map.cuda()
            self.gaussians.active_sh_degree_t = active_sh_degree_t
            cubemap_weights_path = os.path.join(self.model_path, "cubemap/iteration_14000/cubemap.pth")
            self.gaussians.brdf_mlp = load_env(torch.load(cubemap_weights_path))
        else:
            if self.loaded_iter:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"))
            else:
                #self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
                self.gaussians.create_from_multi_pcd(args.source_path, tgh, self.cameras_extent, tgh.time_duration)
                #self.tgh.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, opt, tgh):
        #torch.save((self.gaussians.capture(), iteration), self.model_path + "/chkpnt1_" + str(iteration) + ".pth")
        #tgh.create_from_gaussians(gaussians)
        torch.save((tgh.capture(self.gaussians, opt), iteration), self.model_path + "/tgh_chkpnt" + str(iteration) + ".pth")
        brdf_mlp_path = os.path.join(self.model_path, f"brdf_mlp/iteration_{iteration}/brdf_mlp.hdr")
        mkdir_p(os.path.dirname(brdf_mlp_path))
        save_env_map(brdf_mlp_path, self.gaussians.brdf_mlp)
        cubemap_path = os.path.join(self.model_path, "cubemap/iteration_{}".format(iteration))
        os.makedirs(cubemap_path, exist_ok=True)
        torch.save(self.gaussians.brdf_mlp.base, os.path.join(cubemap_path, 'cubemap.pth'))

    def getTrainCameras(self, scale=1.0):
        return CameraDataset(self.train_cameras[scale].copy(), self.white_background)
        
    def getTestCameras(self, scale=1.0):
        return CameraDataset(self.test_cameras[scale].copy(), self.white_background)