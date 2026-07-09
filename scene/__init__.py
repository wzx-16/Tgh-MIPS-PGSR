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

import gc
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

def _infer_checkpoint_id(path, fallback=0):
    basename = os.path.basename(path or "")
    for prefix in ("gaussian", "tgh", "local"):
        if not basename.startswith(prefix):
            continue
        suffix = basename[len(prefix):]
        if "_chkpnt" not in suffix:
            continue
        candidate = suffix.split("_chkpnt", 1)[0]
        if candidate.isdigit():
            return int(candidate)
    return fallback


def _split_gaussian_checkpoint_payload(payload):
    if isinstance(payload, dict) and "gaussians" in payload:
        return payload["gaussians"], payload.get("local_gaussians")
    return payload, None

class Scene:

    gaussians : GaussianModel
    local_gaussians : GaussianModel
    tgh : TemperalGaussianHierarchy

    def __init__(self, args : ModelParams, gaussians : GaussianModel, tgh : TemperalGaussianHierarchy, local_gaussians: GaussianModel = None, load_iteration=None, shuffle=True, resolution_scales=[1.0], num_pts=100_000, num_pts_ratio=1.0, time_duration=None, render_only=False, skip_render=False, eid=0):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.local_gaussians = local_gaussians
        self.local_gaussians_loaded = False
        self.loaded_gaussian_checkpoint = False
        self.loaded_tgh_checkpoint = False
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

        skip_source_point_cloud = bool(args.loaded_pth) or render_only

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, num_pts_ratio=num_pts_ratio)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, num_pts=num_pts, time_duration=time_duration, extension=args.extension, num_extra_pts=args.num_extra_pts, frame_ratio=args.frame_ratio, dataloader=args.dataloader, render=skip_source_point_cloud, frame_filter=getattr(args, "frame_filter", ""))
        elif os.path.exists(os.path.join(args.source_path, "calibration_full.json")):
            print("Found calibration_full.json file, assuming THU data set!")
            scene_info = sceneLoadTypeCallbacks["THU"](args.source_path, args.white_background, num_pts=num_pts, time_duration=time_duration, num_extra_pts=args.num_extra_pts, frame_ratio=args.frame_ratio, dataloader=args.dataloader)
        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            print("Found dataset.json, assuming nerfies dataset")
            scene_info = sceneLoadTypeCallbacks["Nerfies"](args.source_path, True)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            if not render_only and scene_info.ply_path is not None:
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
            # Gaussian-only checkpoints are used for training resume. Older TGH checkpoints
            # remain loadable for existing rendered/evaluation artifacts.
            checkpoint_basename = os.path.basename(args.loaded_pth)
            is_tgh_checkpoint = checkpoint_basename.startswith("tgh")
            checkpoint_map_location = "cpu" if is_tgh_checkpoint else ("cuda:0" if torch.cuda.is_available() else "cpu")
            checkpoint_args, _ = torch.load(args.loaded_pth, map_location=checkpoint_map_location, weights_only=False)
            embedded_local_args = None
            if not is_tgh_checkpoint:
                checkpoint_args, embedded_local_args = _split_gaussian_checkpoint_payload(checkpoint_args)
            if not (checkpoint_basename.startswith("gaussian") or is_tgh_checkpoint):
                is_tgh_checkpoint = (
                    isinstance(checkpoint_args, tuple)
                    and len(checkpoint_args) == 13
                    and isinstance(checkpoint_args[4], list)
                )
            if is_tgh_checkpoint:
                (active_sh_degree, spatial_lr_scale, env_map, active_sh_degree_t) = self.tgh.restore(model_args=checkpoint_args, training_args=None)
                self.gaussians.active_sh_degree = active_sh_degree
                self.gaussians.spatial_lr_scale = spatial_lr_scale
                self.gaussians.rot_4d = self.tgh.rot_4d
                if env_map is not None:
                    self.gaussians.env_map = env_map.cuda()
                self.gaussians.active_sh_degree_t = active_sh_degree_t
                self.loaded_tgh_checkpoint = True
            else:
                self.gaussians.restore(checkpoint_args, training_args=None)
                self.loaded_gaussian_checkpoint = True
            del checkpoint_args
            gc.collect()
            # cubemap_weights_path = os.path.join(self.model_path, "cubemap/iteration_60000/cubemap.pth")
            # #cubemap_weights_2_path = os.path.join(self.model_path, "cubemap_2/iteration_20000/cubemap.pth")
            # self.gaussians.brdf_mlp = load_env(torch.load(cubemap_weights_path))
            # self.gaussians.light_mlp = torch.load(args.model_path + 'light/iteration_'+str(60000)+'/light_mlp.pt', weights_only=False)
            # self.gaussians.dir_encoding = torch.load(args.model_path + 'dir/iteration_'+str(60000)+'/dir_encoding.pt', weights_only=False)

            iteration = args.loaded_pth.split("chkpnt")[-1].split(".pth")[0]
            try:
                self.loaded_iter = int(iteration)
            except ValueError:
                self.loaded_iter = iteration
            load_eid = _infer_checkpoint_id(args.loaded_pth, eid)
            print("loading env maps from id ", load_eid, " at iteration ", iteration)
            cubemap_weights_path = os.path.join(self.model_path, f"cubemap{load_eid}/iteration_{iteration}/cubemap.pth")
            #cubemap_weights_2_path = os.path.join(self.model_path, "cubemap_2/iteration_20000/cubemap.pth")
            self.gaussians.brdf_mlp = load_env(torch.load(cubemap_weights_path, weights_only=False))
            light_path = os.path.join(self.model_path, f"light{load_eid}", f"iteration_{iteration}")
            self.gaussians.light_mlp = torch.load(os.path.join(light_path, "light_mlp.pt"), weights_only=False)
            self.gaussians.light_mlp_2 = torch.load(os.path.join(light_path, "light_mlp2.pt"), weights_only=False)
            print("Loaded light_mlp:", self.gaussians.light_mlp)
            print("Loaded light_mlp param size: {}".format(sum(p.numel() for p in self.gaussians.light_mlp.parameters())))
            # self.gaussians.light_mlp2 = torch.load(args.model_path + f'light{eid}/iteration_'+ iteration +'/light_mlp2.pt', weights_only=False)
            dir_path = os.path.join(self.model_path, f"dir{load_eid}", f"iteration_{iteration}")
            self.gaussians.dir_encoding = torch.load(os.path.join(dir_path, "dir_encoding.pt"), weights_only=False)

            if self.local_gaussians is not None:
                local_map_location = getattr(self.local_gaussians, "device", "cuda")
                if isinstance(local_map_location, torch.device):
                    local_map_location = str(local_map_location)
                if isinstance(local_map_location, str) and local_map_location.startswith("cuda") and not torch.cuda.is_available():
                    local_map_location = "cpu"

                if embedded_local_args is not None:
                    self.local_gaussians.restore(embedded_local_args, training_args=None)
                    self.local_gaussians_loaded = True
                    print("Loaded embedded local Gaussians from Gaussian checkpoint.")
                else:
                    local_ckpt_candidates = [
                        os.path.join(self.model_path, f"local{load_eid}_chkpnt{iteration}.pth"),
                        os.path.join(self.model_path, f"local{eid}_chkpnt{iteration}.pth"),
                        os.path.join(self.model_path, f"local_chkpnt{iteration}.pth"),
                    ]
                    for local_ckpt_path in local_ckpt_candidates:
                        if not os.path.exists(local_ckpt_path):
                            continue
                        try:
                            local_model_args, _ = torch.load(local_ckpt_path, map_location=local_map_location, weights_only=False)
                            self.local_gaussians.restore(local_model_args, training_args=None)
                            self.local_gaussians_loaded = True
                            print(f"Loaded local Gaussians from {local_ckpt_path}")
                            break
                        except Exception as exc:
                            print(f"[Scene] Failed to load local Gaussians from {local_ckpt_path}: {exc}")

            # self.gaussians.dir_encoding2 = torch.load(args.model_path + f'dir{eid}/iteration_'+ iteration +'/dir_encoding2.pt', weights_only=False)
            #self.gaussians.brdf_mlp_2 = load_env(torch.load(cubemap_weights_2_path))
        elif skip_render:
            pass
        else:
            if self.loaded_iter:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"))
            else:
                #self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
                scratch_max_points = num_pts if num_pts < 500_000 else None
                self.gaussians.create_from_multi_pcd(args.source_path, tgh, self.cameras_extent, tgh.time_duration, max_points=scratch_max_points, frame_filter=getattr(args, "frame_filter", ""))
                #pass
                #self.tgh.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, opt, tgh, id):
        torch.save((self.gaussians.capture(), iteration), self.model_path + f"/gaussian{id}_chkpnt" + str(iteration) + ".pth")
        if self.local_gaussians is not None:
            torch.save((self.local_gaussians.capture(), iteration), self.model_path + f"/local{id}_chkpnt" + str(iteration) + ".pth")
        brdf_mlp_path = os.path.join(self.model_path, f"brdf_mlp{id}/iteration_{iteration}/brdf_mlp.hdr")
        #brdf_mlp_2_path = os.path.join(self.model_path, f"brdf_mlp_2/iteration_{iteration}/brdf_mlp.hdr")
        mkdir_p(os.path.dirname(brdf_mlp_path))
        #mkdir_p(os.path.dirname(brdf_mlp_2_path))
        save_env_map(brdf_mlp_path, self.gaussians.brdf_mlp)
        #save_env_map(brdf_mlp_2_path, self.gaussians.brdf_mlp_2)
        cubemap_path = os.path.join(self.model_path, f"cubemap{id}/iteration_{iteration}")
        light_path = os.path.join(self.model_path, f"light{id}/iteration_{iteration}")
        dir_path = os.path.join(self.model_path, f"dir{id}/iteration_{iteration}")
        #cubemap_2_path = os.path.join(self.model_path, "cubemap_2/iteration_{}".format(iteration))
        os.makedirs(cubemap_path, exist_ok=True)
        os.makedirs(light_path, exist_ok=True)
        os.makedirs(dir_path, exist_ok=True)
        #os.makedirs(cubemap_2_path, exist_ok=True)
        torch.save(self.gaussians.brdf_mlp.base, os.path.join(cubemap_path, 'cubemap.pth'))
        torch.save(self.gaussians.light_mlp, os.path.join(light_path,'light_mlp.pt'))
        torch.save(self.gaussians.light_mlp_2, os.path.join(light_path,'light_mlp2.pt'))
        torch.save(self.gaussians.dir_encoding, os.path.join(dir_path,'dir_encoding.pt'))
        #torch.save(self.gaussians.brdf_mlp_2.base, os.path.join(cubemap_2_path, 'cubemap.pth'))

    def getTrainCameras(self, scale=1.0):
        return CameraDataset(self.train_cameras[scale].copy(), self.white_background)
        
    def getTestCameras(self, scale=1.0):
        return CameraDataset(self.test_cameras[scale].copy(), self.white_background)
