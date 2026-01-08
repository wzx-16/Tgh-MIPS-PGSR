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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import os
from PIL import Image
import torch

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height# cam_info.image.size

    if args.resolution in [1, 2, 3, 4, 6, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        scale = resolution_scale * args.resolution
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    
    cx = cam_info.cx / scale
    cy = cam_info.cy / scale
    fl_y = cam_info.fl_y / scale
    fl_x = cam_info.fl_x / scale
    
    loaded_mask = None
    if not args.dataloader:
        #resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        if cam_info.image is not None:
            resized_image = torch.from_numpy(np.array(cam_info.image)) / 255.0
            if len(resized_image.shape) == 3:
                resized_image_rgb = resized_image.permute(2, 0, 1)
            else:
                resized_image_rgb = resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
            gt_image = resized_image_rgb[:3, ...]
        else:
            gt_image = None
            resized_image_rgb = None

        # mask_path = os.path.join(cam_info.image_path, os.path.joint("../mattings", cam_info.image_name.split("_")[0]))
        # mask_name = cam_info.image_name.split("_")[-1].split(".")[0] + ".png"
        # with Image.open(os.path.join(mask_path, mask_name)) as image_load:
        #     loaded_mask_PIL = image_load.resize((int(orig_w / 2), int(orig_h / 2)))
        # loaded_mask = torch.from_numpy(np.array(loaded_mask_PIL)) / 255.0
        # loaded_mask = loaded_mask.permute(2, 0, 1)
        if resized_image_rgb is not None and resized_image_rgb.shape[0] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]
    else:
        gt_image = cam_info.image
        #loaded_mask = cam_info.loaded_mask
        # mask_path = "/" + os.path.join(os.path.join(*cam_info.image_path.split("/")[0:-2]), os.path.join("mattings", cam_info.image_name.split("_")[0]))
        # mask_name = cam_info.image_name.split("_")[-1].split(".")[0] + ".png"
        # with Image.open(os.path.join(mask_path, mask_name)) as image_load:
        #     loaded_mask_PIL = image_load.resize((int(orig_w), int(orig_h)))
        # loaded_mask = torch.from_numpy(np.array(loaded_mask_PIL)) / 255.0
        #loaded_mask = loaded_mask.permute(2, 0, 1)
    
    if cam_info.depth is not None:
        depth = PILtoTorch(cam_info.depth, resolution) * 255 / 10000
    else:
        depth = None

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, 
                  timestamp=cam_info.timestamp, W=resolution[0], H=resolution[1],
                  cx=cx, cy=cy, fl_x=fl_x, fl_y=fl_y, depth=depth, resolution=resolution, image_path=cam_info.image_path,
                  meta_only=args.dataloader
                  )

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
