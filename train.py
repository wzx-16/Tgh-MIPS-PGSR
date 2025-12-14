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
import os
import random
import torch
from torch import nn
from utils.loss_utils import get_img_grad_weight, get_img_grad_weight_grey, get_img_grad_weight_avg, l1_loss, ssim, msssim
from gaussian_renderer import render, render_3d_pgsr_anti
import sys
from scene import Scene, GaussianModel, TemperalGaussianHierarchy
from utils.general_utils import safe_state, safe_normalize, reflect
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, easy_cmap
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import make_grid
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader
import cv2
# import copy_and_cat_engine
import lpips
import gc
from datetime import datetime
from PIL import Image
import time
import torch.multiprocessing
from torchvision import transforms
import torchvision

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from transformers import pipeline
from diffusers import DiffusionPipeline
import diffusers

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint, debug_from,
             gaussian_dim, time_duration, num_pts, num_pts_ratio, rot_4d, force_sh_3d, batch_size):
    
    # import os, torch, torch.nn as nn
    # print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cudnn:", torch.backends.cudnn.version())
    # print("cuda_available:", torch.cuda.is_available(), "cudnn_available:", torch.backends.cudnn.is_available())
    # print("device_count:", torch.cuda.device_count(), "name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # # Sanity conv on your GPU
    # if torch.cuda.is_available():
    #     x = torch.randn(2, 3, 224, 224, device="cuda", dtype=torch.float32)
    #     m = nn.Conv2d(3, 16, 3, padding=1).cuda().float()
    #     y = m(x)
    #     print("sanity conv OK:", y.shape)
    #torch.autograd.set_detect_anomaly(True)

    if dataset.frame_ratio > 1:
        time_duration = [time_duration[0] / dataset.frame_ratio,  time_duration[1] / dataset.frame_ratio]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    first_iter = 0
    # tb_writer = prepare_output_and_logger(dataset)
    tgh = TemperalGaussianHierarchy(dataset.sh_degree, 9, 10,  gaussian_dim=gaussian_dim, time_duration=time_duration, rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0, device=device, opt=opt)
    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0, device='cuda')
    scene = Scene(dataset, gaussians, tgh, num_pts=num_pts, num_pts_ratio=num_pts_ratio, time_duration=time_duration)
    
    #checkpoint = './output/N3V/tao/tgh_chkpnt5000.pth'
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        scene.tgh.create_from_gaussians(gaussians)
        gaussian_init_flag = True
    else:
        gaussian_init_flag = False
        gaussians.training_setup(opt)
    #print("test5")
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    best_psnr = 0.0
    ema_loss_for_log = 0.0
    ema_l1loss_for_log = 0.0
    ema_ssimloss_for_log = 0.0
    lambda_all = [key for key in opt.__dict__.keys() if key.startswith('lambda') and key!='lambda_dssim']
    for lambda_name in lambda_all:
        vars()[f"ema_{lambda_name.replace('lambda_','')}_for_log"] = 0.0
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
        
    # if pipe.env_map_res:
    #     env_map = nn.Parameter(torch.zeros((3,pipe.env_map_res, pipe.env_map_res),dtype=torch.float, device="cuda").requires_grad_(True))
    #     env_map_optimizer = torch.optim.Adam([env_map], lr=opt.feature_lr, eps=1e-15)
    # else:
    env_map = None
        
    gaussians.env_map = env_map

    #scene.tgh.create_from_gaussians(gaussians, opt)
    # gaussian_init_flag = False
    training_dataset = scene.getTrainCameras()
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=12 if dataset.dataloader else 0, collate_fn=lambda x: x, drop_last=True, pin_memory=True)
    #print("test6")
    iteration = first_iter
    fn_lpips = lpips.LPIPS(net='vgg').cuda().eval()
    depth_guidance_checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
    #depth_guidance_checkpoint = "LiheYoung/depth-anything-base-hf"
    #pipe_depth = pipeline("depth-estimation", model=depth_guidance_checkpoint, device="cuda")
    #pipe_depth.model.eval()

    #pipe_normal = DiffusionPipeline.from_pretrained("GonzaloMG/stable-diffusion-e2e-ft-normals",custom_pipeline="/root/autodl-tmp/projects/marigold-e2e-ft-normals/pipeline.py", trust_remote_code=True, local_files_only=True).to("cuda")
    #pipe_normal = DiffusionPipeline.from_pretrained("GonzaloMG/stable-diffusion-e2e-ft-normals",custom_pipeline="GonzaloMG/marigold-e2e-ft-normals",).to("cuda")
    #pipe_normal = diffusers.MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v1-1", variant="fp16", torch_dtype=torch.float16).to("cuda")
    # vis = pipe_normal.image_processor.visualize_normals(normals.prediction, flip_x=False)
    # pipe_normal.eval()
    # def lpips_tiled(im_chw, gt_chw, tile=512, overlap=64):
    #     im = im_chw.to('cuda', dtype=torch.float32)
    #     gt = gt_chw.to('cuda', dtype=torch.float32).detach()
    #     _,_,H,W = im.shape
    #     step = tile - overlap
    #     total = 0.0
    #     area  = 0
    #     for y in range(0, H, step):
    #         for x in range(0, W, step):
    #             y1, x1 = min(y+tile, H), min(x+tile, W)
    #             d = fn_lpips(im[..., y:y1, x:x1], gt[..., y:y1, x:x1], normalize=True)
    #             a = (y1-y)*(x1-x)
    #             total += d*a
    #             area += a
    #     return total/area
    #print("test7")
    while iteration < opt.iterations + 1:
        for batch_data in training_dataloader:
            #train_start = time.time()
            iteration += 1
            # if iteration > 20:
            #     exit()

            iter_start.record()
            gaussians.update_learning_rate(iteration)
            
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % opt.sh_increase_interval == 0:
                gaussians.oneupSHdegree()
                
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            
            batch_point_grad = []
            batch_visibility_filter = []
            batch_radii = []
            
            #start_t = time.time()
            for batch_idx in range(batch_size):
                gt_image, loaded_mask, viewpoint_cam, n_gt, predicted_normal, predicted_depth = batch_data[batch_idx]
                #gaussians.set_current_timestamp(viewpoint_cam.timestamp)
                if gaussian_init_flag:
                    #cpu_to_cuda_start = time.time()
                    #put_gaussians_start = time.time()

                    #scene.tgh.put_current_related_gaussians(viewpoint_cam.timestamp, gaussians)
                    gaussians.set_current_timestamp(viewpoint_cam.timestamp)

                    #pass
                    # put_gaussians_end = time.time()
                    # torch.cuda.synchronize()
                    # print(f"put gaussians time{put_gaussians_end - put_gaussians_start:.6f}second")
                    #torch.cuda.synchronize()
                    #cpu_to_cuda_end = time.time()
                    #print(f" cpu to cuda time: {cpu_to_cuda_end - cpu_to_cuda_start:.6f} seconds")
                else:
                    gaussians.set_current_timestamp(viewpoint_cam.timestamp)
                #render_start = time.time()
                if iteration < 3000:
                    with Image.open(viewpoint_cam.image_path.replace("images", "images_delight")) as image_load:
                        im_data = np.array(image_load.convert("RGBA"))#[100:-100, 100:-100]
                    norm_data = im_data / 255.0
                    arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg_color * (1 - norm_data[:, :, 3:4])
                    image_load = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
                    image_load = np.array(image_load)
                    #image_load = cv2.resize(image_load, (image_load.shape[1]//2, image_load.shape[0]//2), interpolation=cv2.INTER_LINEAR)
                    image_load = torch.from_numpy(image_load) / 255.0
                    resized_image_rgb = image_load.permute(2, 0, 1)
                    viewpoint_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
                    gt_image = viewpoint_image
                gt_image = gt_image.cuda()
                viewpoint_cam = viewpoint_cam.cuda()
                #loaded_mask = loaded_mask.cuda()
                # sky = (1 - loaded_mask) > 1 - 1e-6
                # origin_gt = gt_image
                #random_color = torch.zeros_like(gt_image, device = "cuda")
        
                #channel_idx = np.random.randint(0,3)
                #print("random color", channel_idx)
                #random_color[channel_idx, :, :] = 1.0
                #gt_image[:, sky[0]] = random_color[:, sky[0]]
                #render_start = time.time()
                # copy_and_cat_engine.waitGroupCompletion(0, 0)
                # render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                # #torch.cuda.synchronize()
                # #render_end = time.time()
                # #print(f"render time: {render_end - render_start:.6f} seconds")
                # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                # depth = render_pkg["depth"]
                # alpha = render_pkg["alpha"]
                
                viewpoint_cam = viewpoint_cam.cuda()
                # _, gpu_mask = t_tree_model.find_t_batch([viewpoint_cam.timestamp])
                
                #xyz = gaussians.get_xyz + gaussians.get_velocity * (viewpoint_cam.timestamp - gaussians.get_t) / (gaussians.get_sigma_t + 1)
                #xyz = gaussians.get_xyz + gaussians.get_velocity * (viewpoint_cam.timestamp - gaussians.get_t)
                time_range = viewpoint_cam.timestamp - gaussians.get_t
                #time_range_offset = torch.abs(time_range) + 0.5
                # time_range2 = time_range_offset * time_range_offset
                # time_range3 = time_range_offset**2 * time_range
                #time_range2 = time_range**2
                #time_range3 = time_range**3
                # if iteration >= 3000 and iteration < 5000:
                #     xyz = gaussians.get_xyz.detach() + gaussians.get_velocity.detach() * time_range / (gaussians.get_sigma_t + 1)
                xyz = gaussians.get_xyz + gaussians.get_velocity * time_range / (gaussians.get_sigma_t + 1)
                if iteration >= 3000 and iteration < 5000:
                    xyz = xyz.detach()
                #xyz = gaussians.get_xyz + (gaussians.get_velocity * time_range + gaussians.get_velocity2 * time_range2 + gaussians.get_velocity3 * time_range3) / (gaussians.get_sigma_t + 1)# + gaussians.get_velocity2 * time_range2 + gaussians.get_velocity3 * time_range3
                #rot = gaussians.get_rotation + gaussians.get_rot_velocity * (viewpoint_cam.timestamp - gaussians.get_t)
                # xyz = gaussians.get_xyz + gaussians.get_velocity * (viewpoint_cam.timestamp - gaussians.get_t) / (gaussians.get_sigma_t.detach() + 1)
                mt = gaussians.get_marginal_t(timestamp=viewpoint_cam.timestamp)
                opacity = gaussians.get_opacity * mt
                # plt.hist(opacity[t_tree_model.t_tree[0].shape[0]:].detach().cpu().numpy(), bins=100, range=(0, 1))
                # plt.show()
                shs = gaussians.get_features
                # ma = torch.ones_like(opacity[..., 0], dtype=torch.bool, device=opacity.device)
                ma = (mt > 0.05).squeeze()
                # plt.hist(opacity[ma].detach().cpu().numpy(), bins=100, range=(0, 1))
                # plt.show()
                #background = torch.rand(3, device="cuda")
                # sky_mask = (1 - loaded_mask) > 1 - 2e-2
                # random_color = torch.zeros_like(gt_image, device = "cuda")
                # random_color[0, :, :] = background[0]
                # random_color[1, :, :] = background[1]
                # random_color[2, :, :] = background[2]
                # gt_image[:, sky_mask[0]] = random_color[:, sky_mask[0]]
                # sky_mask_percentage = 1 - loaded_mask
                # random_color = torch.zeros_like(gt_image, device = "cuda")
                # random_color[0, :, :] = background[0]
                # random_color[1, :, :] = background[1]
                # random_color[2, :, :] = background[2]
                # gt_image = gt_image * loaded_mask + random_color * sky_mask_percentage
                # print("gaussiansize")
                # print(xyz.size())
                # print(opacity.size())
                gaussians.brdf_mlp.build_mips()
                #gaussians.brdf_mlp_2.build_mips()
                view_pos = viewpoint_cam.camera_center.repeat(gaussians.get_opacity.shape[0], 1) 
                d_viewdir_normalized = safe_normalize(view_pos - xyz)
                normal = gaussians.get_normal(viewpoint_cam.camera_center, xyz)
                normal = normal + gaussians.get_delta_normal
                reflvec = safe_normalize(reflect(d_viewdir_normalized, normal))
                dir_pp = (xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1)).detach()
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                render_pkg = render_3d_pgsr_anti(viewpoint_cam, xyz, None, opacity, gaussians.active_sh_degree, 
                                    gaussians.get_scaling, gaussians.get_rotation, background, shs=shs, mask=ma, max_sh_channels=gaussians.max_sh_degree,normal =normal, reflect=reflvec, dir_pp=dir_pp_normalized, pc=gaussians, iteration=iteration)
                # d_viewdir_normalized = safe_normalize(view_pos - )
                # reflvec = safe_normalize(reflect(d_viewdir_normalized, render_pkg["rendered_normal"]))
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                
                # render_end = time.time()
                # torch.cuda.synchronize()
                # print(f"render time {render_end - render_start:.6f} second")
                if iteration % 100 == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite("./test/debug_render_{}.jpg".format(timestamp + "_" + viewpoint_cam.image_name), np.hstack(((gt_image.clip(min=0, max=1).squeeze().permute(1,2,0).detach().cpu().numpy()[..., [2,1,0]] * 255).astype(np.uint8), (image.clip(min=0, max=1).squeeze().permute(1,2,0).detach().cpu().numpy()[..., [2,1,0]] * 255).astype(np.uint8))))
                
                #loss_start = time.time()
                # Loss
                Ll1 = l1_loss(image, gt_image)
                Lssim = 1.0 - ssim(image, gt_image)
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim
                # patch_size = 1024#48 * 14
                # random_v = torch.randint(0, gt_image.shape[-2] - patch_size, (1,))
                # random_u = torch.randint(0, gt_image.shape[-1] - patch_size, (1,))
                # lp = fn_lpips(image[None, :, random_v:random_v+patch_size, random_u:random_u+patch_size], gt_image[None, :, random_v:random_v+patch_size, random_u:random_u+patch_size], normalize=True)
                # print("test8")
                # torch.cuda.empty_cache()
                # gc.collect()
                #print("image", image.shape, image.dtype, image.device, "gt", gt_image.shape)
                lp = fn_lpips(image[None], gt_image[None], normalize=True)
                #lp = lpips_tiled(image[None], gt_image[None])
                #print("test9")
                # gt_image_resize = torch.nn.functional.interpolate(gt_image[None], size=(1960//2, 3640//2), mode='bilinear')
                # image_resize = torch.nn.functional.interpolate(image[None], size=(1960//2, 3640//2), mode='bilinear')
                # lp_resize = fn_lpips(image_resize, gt_image_resize, normalize=True)
                loss = loss + 0.01 * lp.mean()
                #print("loss 1", loss)
                weight_conf = 1.0 - get_img_grad_weight(gt_image)
                decay_weight = get_decay_weight(15000, 30000, iteration)
                if iteration < 20000:
                    with torch.no_grad():
                        # to_pil_image = transforms.ToPILImage()
                        # gt_pil = to_pil_image(gt_image)
                        # sgt_depth = pipe_depth(gt_pil)
                        # predicted_depth = sgt_depth["predicted_depth"].cuda()
                        predicted_depth = torch.from_numpy(predicted_depth).cuda()
                        #print("predicted depth", predicted_depth.shape)
                    avg_diff_pred = torch.mean(torch.abs(predicted_depth - predicted_depth.median()))
                    depth_norm = (predicted_depth - predicted_depth.median()) / avg_diff_pred
                    render_depth = render_pkg["depth"]
                    #print("render depth", render_depth.shape)
                    avg_diff_render = torch.mean(torch.abs(render_depth - render_depth.median()))
                    render_depth_norm = (render_depth - render_depth.median()) / avg_diff_render
                    depth_grad = (1 - get_img_grad_weight_grey(predicted_depth)).detach()
                    if iteration < 1000:
                        pass
                    # elif iteration < 6000:
                    #     loss += 0.01 * torch.abs(depth_norm + render_depth_norm).mean()
                    elif iteration < 20000:
                        loss += 0.02 * (torch.abs((depth_norm + render_depth_norm))).mean()
                    elif iteration < 15000:
                        loss += 0.3 * (depth_grad * torch.abs((depth_norm + render_depth_norm))).mean()
                    elif iteration < 30000:
                        loss += 0.3 * decay_weight * (depth_grad * torch.abs((depth_norm + render_depth_norm))).mean()
                    if iteration % 100 == 1:
                        predicted_depth_image = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
                        render_depth_image = (render_depth - render_depth.min()) / (render_depth.max() - render_depth.min())
                        torchvision.utils.save_image(render_depth_image, "render_depth.png")
                        torchvision.utils.save_image(predicted_depth_image, "predicted_depth.png")

                if iteration < 20000:
                    with torch.no_grad():
                        # to_pil_image = transforms.ToPILImage()
                        # gt_pil = to_pil_image(gt_image)
                        # sgt_normal = pipe_normal(gt_pil)
                        #image_test = diffusers.utils.load_image("https://gonzalomartingarcia.github.io/diffusion-e2e-ft/static/lego.jpg")
                        #normal_test = pipe_normal(image_test)
                        #pipe_normal.image_processor.visualize_normals(normal_test.prediction)[0].save("normals.png")
                        #print("sgt_normal", sgt_normal["prediction"].shape)
                        predicted_normal = torch.from_numpy(predicted_normal).permute(0,3,1,2).squeeze(0).cuda()
                        #predicted_normal = predicted_normal.reshape(3, -1)
                        #_, H, W = gt_image.shape
                        #pred_normal_rot = (torch.from_numpy(viewpoint_cam.R).to(torch.float32).cuda() @ predicted_normal).reshape(3, H, W)
                        pred_normal_rot = predicted_normal
                        pred_normal_rot[0, :, :] = -pred_normal_rot[0, :, :]
                        pred_normal_rot[1, :, :] = -pred_normal_rot[1, :, :]
                        pred_normal_rot[2, :, :] = -pred_normal_rot[2, :, :]
                        #print("sgt_normal", predicted_normal.shape)
                    #normal_norm = (predicted_normal - predicted_normal.mean()) / predicted_normal.std()
                    normal_norm = torch.nn.functional.normalize(pred_normal_rot, dim=0, eps=1e-6)
                    render_normal = render_pkg["rendered_normal"]
                    #render_normal_norm = (render_normal - render_normal.mean()) / render_normal.std()
                    render_normal_norm = torch.nn.functional.normalize(render_normal, dim=0, eps=1e-6)
                    # print("render normal", render_normal_norm.shape)
                    # print("predict normal", normal_norm.shape)
                    normal_grad = get_img_grad_weight_avg(normal_norm)
                    render_normal_grad = get_img_grad_weight_avg(render_normal_norm)
                    #print("normal grad", normal_grad.shape)
                    #print("render normal grad", render_normal_grad.shape)
                    # with torch.no_grad():
                    #     to_pil_image = transforms.ToPILImage()
                    #     gt_pil = to_pil_image(gt_image)
                    #     sgt_depth = pipe_depth(gt_pil)
                    #     predicted_depth = sgt_depth["predicted_depth"].cuda()
                    #print("depth size", predicted_depth.shape)
                    #depth_grad = get_img_grad_weight_grey(predicted_depth)
                    #print("depth grad", depth_grad.shape)
                    if iteration < 1000:
                        pass
                    elif iteration < 20000:
                        loss += 0.1 * ((1 - (render_normal_norm * normal_norm).sum(dim=0))).mean()
                    elif iteration < 15000:
                        loss += 0.02 * (depth_grad * (1 - (render_normal_norm * normal_norm).sum(dim=0))).mean()
                        #loss += 0.02 * (depth_grad * ((normal_grad - render_normal_grad).abs().sum(dim=0))).mean()
                    elif iteration < 30000:
                        loss += 0.02 * decay_weight * (depth_grad * (1 - (render_normal_norm * normal_norm).sum(dim=0))).mean()
                        #loss += 0.01 * decay_weight * (depth_grad * ((normal_grad - render_normal_grad).abs().sum(dim=0))).mean()

                    if iteration >= 20000 and iteration < 15000:
                        loss += 0.02 * (depth_grad * ((normal_grad - render_normal_grad).abs().sum(dim=0))).mean()
                    elif iteration >= 20000:
                        loss += 0.02 * decay_weight * (depth_grad * ((normal_grad - render_normal_grad).abs().sum(dim=0))).mean()
                    if iteration % 100 == 1:
                    #predicted_normal_image = (predicted_normal - predicted_normal.min()) / (predicted_normal.max() - predicted_normal.min())
                    #render_normal_image = (render_normal - render_normal.min()) / (render_normal.max() - render_normal.min())
                        # render_normal_norm[0, 0:30, 0:30] = 0
                        # render_normal_norm[1, 0:30, 0:30] = 1
                        # render_normal_norm[2, 0:30, 0:30] = 0
                        torchvision.utils.save_image((render_normal_norm + 1) / 2, "render_normal.png")
                        torchvision.utils.save_image((normal_norm + 1) / 2, "predicted_normal.png")
                #print("loss 2", loss)
                #depth_image = depth_image.detach().cpu().numpy() * 255
                # depth_image = depth_image.detach().cpu().numpy() * 255
                # depth_image = Image.fromarray(depth_image.astype("uint8"))
                # depth_image.save("depth.png")
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # cv2.imwrite("./test/debug_render_depth_{}.jpg".format(timestamp + "_" + viewpoint_cam.image_name), (depth_image.clip(min=0, max=1).squeeze().detach().cpu().numpy()[..., [2,1,0]] * 255).astype(np.uint8))
                
                #print("test10")
                # loss = loss + 0.01 * lp.mean() + 0.01 * lp_resize.mean()
                alpha = render_pkg["alpha"]
                #print("test11")
                # ###### opa mask Loss ######
                if opt.lambda_opa_mask > 0:
                    o = alpha.clamp(1e-6, 1-1e-6)
                    # mask_path = "/" + os.path.join(os.path.join(*viewpoint_cam.image_path.split("/")[0:-2]), os.path.join("mattings", viewpoint_cam.image_name.split("_")[0]))
                    # mask_name = viewpoint_cam.image_name.split("_")[-1].split(".")[0] + ".png"
                    # with Image.open(os.path.join(mask_path, mask_name)) as image_load:
                    #     loaded_mask_PIL = image_load.resize((1500, 2000))
                    #loaded_mask = torch.from_numpy(np.array(loaded_mask_PIL)).unsqueeze(0).cuda() / 255.0
                    #sky = 1 - viewpoint_cam.gt_alpha_mask
                    # print(o.shape)
                    # print(loaded_mask.shape)
                    sky = 1 - loaded_mask
                    # sky = torch.ones_like(gt_image[:1])
                    # sky[torch.linalg.norm(gt_image, dim=0, keepdim=True)>0] = 0.0
                    # sky[torch.linalg.norm(gt_image, dim=0, keepdim=True)==0] = 1.0

                    Lopa_mask = (- sky * torch.log(1 - o)).mean()

                    # lambda_opa_mask = opt.lambda_opa_mask * (1 - 0.99 * min(1, iteration/opt.iterations))
                    lambda_opa_mask = opt.lambda_opa_mask
                    loss = loss + lambda_opa_mask * Lopa_mask
                # ###### opa mask Loss ######
                
                # ###### rigid loss ######
                # if opt.lambda_rigid > 0:
                #     k = 20
                #     # cur_time = viewpoint_cam.timestamp
                #     # _, delta_mean = gaussians.get_current_covariance_and_mean_offset(1.0, cur_time)
                #     xyz_mean = gaussians.get_xyz
                #     xyz_cur =  xyz_mean #  + delta_mean
                #     idx, dist = knn(xyz_cur[None].contiguous().detach(), 
                #                     xyz_cur[None].contiguous().detach(), 
                #                     k)
                #     _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)
                #     weight = torch.exp(-100 * dist)
                #     # cur_marginal_t = gaussians.get_marginal_t(cur_time).detach().squeeze(-1)
                #     # marginal_weights = cur_marginal_t[idx] * cur_marginal_t[None,:,None]
                #     # weight *= marginal_weights
                    
                #     # mean_t, cov_t = gaussians.get_t, gaussians.get_cov_t(scaling_modifier=1)
                #     # mean_t_nn, cov_t_nn = mean_t[idx], cov_t[idx]
                #     # weight *= torch.exp(-0.5*(mean_t[None, :, None]-mean_t_nn)**2/cov_t[None, :, None]/cov_t_nn*(cov_t[None, :, None]+cov_t_nn)).squeeze(-1).detach()
                #     vel_dist = torch.norm(velocity[idx] - velocity[None, :, None], p=2, dim=-1)
                #     Lrigid = (weight * vel_dist).sum() / k / xyz_cur.shape[0]
                #     loss = loss + opt.lambda_rigid * Lrigid
                # ########################
                
                # ###### motion loss ######
                # if opt.lambda_motion > 0:
                #     _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)
                #     Lmotion = velocity.norm(p=2, dim=1).mean()
                #     loss = loss + opt.lambda_motion * Lmotion
                # ########################
                #loss += 0.1 * (gaussians.get_scaling[visibility_filter] - 0.2).clip(min=0.0).sum()
                # _, cov_t = gaussians.get_current_cov_and_mean_t()
                cov_t = gaussians.get_sigma_t
                effect_range = torch.sqrt(-2 * torch.log(torch.tensor(0.05, device="cuda")) * cov_t)
                # print(loss, '1')
                #loss += 0.1 * torch.clip(15-effect_range, min=0).mean()
                loss += 0.1 * torch.clip(1/30/2 - effect_range, min=0.0).mean()
                #loss += 0.1 * torch.clip(5 - effect_range, min = 0.0).mean()
                # print(loss, '2')
                #loss += 0.1 * (gaussians.get_opacity[gaussians.get_opacity>0.5] * gaussians.get_opacity[gaussians.get_opacity>0.5].detach() - 0.0).clip(min=0.0).mean()
                # print(loss, '3')
                # depth = torch.where(depth.isnan() | depth.isinf(), torch.zeros_like(depth), depth)
                # loss += (2 - depth[depth < 2]).sum() * 0.01
                # print(loss, '4', depth[depth < 3].isinf().any())
                
                if iteration > 0 and visibility_filter.sum() > 0:
                    scale = gaussians.get_scaling[visibility_filter]
                    sorted_scale, _ = torch.sort(scale, dim=-1)
                    min_scale_loss = sorted_scale[...,0]
                    loss += 100 * min_scale_loss.mean()
                    loss += 0.01 * (gaussians.get_scaling[visibility_filter] - 0.2).clip(min=0.0).sum()
                    #loss += 0.001 * (gaussians.get_velocity / (gaussians.get_sigma_t + 1)).abs().mean()  # encourage velocity to be small
                    #velocity_norm = torch.norm(gaussians.get_velocity.detach(), p=2, dim=1).unsqueeze(1)
                    #velocity_norm2 = torch.norm(gaussians.get_velocity2.detach(), p=2, dim=1).unsqueeze(1)
                    #velocity_norm3 = torch.norm(gaussians.get_velocity3.detach(), p=2, dim=1).unsqueeze(1)
                    #print("velocity_norm", velocity_norm.size())
                    #print("effect_range", effect_range.size())
                    #loss += 0.1 * ((time_duration[1] - time_duration[0]) - (((velocity_norm + velocity_norm2 + velocity_norm3) / 2 + 1) * effect_range)).clip(min=0).mean()
                    #loss += 0.1 * ((velocity_norm2 + velocity_norm3) / (velocity_norm + 1e-6)).mean()

                # single-view loss
                if iteration > 1500:
                    if iteration < 20000:
                        weight = 0.15
                    else:
                        weight = 0.05
                    normal = render_pkg["rendered_normal"]
                    depth_normal = render_pkg["depth_normal"]
                    normal = torch.nn.functional.normalize(normal, dim=0, eps=1e-20)
                    depth_normal = torch.nn.functional.normalize(depth_normal, dim=0, eps=1e-20)
                    #normal_grad = get_img_grad_weight_avg(normal)
                    depth_normal_grad = get_img_grad_weight_avg(depth_normal)
                    
                    if n_gt is not None:
                        n_gt = n_gt.cuda()
                        loss += (((n_gt - normal)).abs().sum(0)).mean() * 0.05
                        # loss += (((n_gt - normal)).abs().sum(0)).mean() * 0.5
                    # if gt_mask is not None:
                    #     gt_mask = gt_mask.cuda()[100:-100, 100:-100]
                    #     if gt_mask.shape[0] + gt_mask.shape[1] > 1:
                    #         # print(gt_mask.shape, gt_image.shape, image.shape)
                    #         loss += (image[:, gt_mask] - gt_image[:, gt_mask]).abs().mean() * 0.1
                        # loss += l1_loss(image * gt_mask[None, 180:-180, 120:-120], gt_image * gt_mask[None, 180:-180, 120:-120])
                    # cam0, t0 = os.path.split(viewpoint_cam.image_path)[-1].split('_')
                    # if int(t0[:-4]) < 10:
                    #     de0 = des[int(cam0[3:])-1]
                    #     de0 = torch.nn.functional.interpolate(de0.squeeze()[None, None], size=(1080, 1920), mode='bilinear')
                    #     loss += (((render_pkg["depth"]-render_pkg["depth"].min())/(render_pkg["depth"].max()-render_pkg["depth"].min())).squeeze() - \
                    #         ((de0-de0.min())/(de0.max()-de0.min())).squeeze()).abs().mean() * 0.1
                    #     print('used')
                        
                    #     # cv2.imwrite("./test/debug_render1.png", (((de0-de0.min())/(de0.max()-de0.min())).clip(min=0, max=1).squeeze()[..., None][..., [0]*3].detach().cpu().numpy() * 255).astype(np.uint8))

                    image_weight = (1.0 - get_img_grad_weight(gt_image))
                    image_weight = (image_weight).clamp(0,1).detach() ** 2
                    if True:
                        # image_weight = erode(image_weight[None,None]).squeeze()
                        #normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
                        normal_loss = weight * ((1 - ((depth_normal * normal).sum(dim=0)))).mean()
                        #normal_grad_loss = 0.01 * (depth_grad * ((render_normal_grad.detach() - depth_normal_grad).abs().sum(dim=0))).mean()
                        #pass
                    else:
                        normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
                    loss += (normal_loss)# + (((normal_image - normal)).abs().sum(0)).mean()
                    #loss += normal_grad_loss
                    loss += (1 - render_pkg["alpha"]).mean() * 0.1  # encourage alpha to be 1
                    if iteration <= 20000:
                        loss += (-opacity * torch.log(opacity)).mean() * 0.1
                    elif iteration <= 15000:
                        loss += (-gaussians.get_opacity * torch.log(gaussians.get_opacity)).mean() * 0.5
                    else:
                        loss += (-gaussians.get_opacity * torch.log(gaussians.get_opacity)).mean() * 0.5 * decay_weight
                        #loss += normal_grad_loss
                    if iteration >= 1000:
                        loss += 0.01 * (gaussians.get_delta_normal**2).mean()
                    # if iteration >= 3000:
                    #     loss += 0.1 * (gaussians.get_specular).mean()
                loss = loss / batch_size
                loss.backward()
                batch_point_grad.append(torch.norm(viewspace_point_tensor.grad[:,:2], dim=-1))
                batch_radii.append(radii)
                batch_visibility_filter.append(visibility_filter)
                # loss_end = time.time()
                # torch.cuda.synchronize()
                # print(f"loss compute time: {loss_end - loss_start:.6f} seconds")
                #scene.tgh.update_from_gaussians(gaussians, opt)

            if batch_size > 1:
                visibility_count = torch.stack(batch_visibility_filter,1).sum(1)
                visibility_filter = visibility_count > 0
                radii = torch.stack(batch_radii,1).max(1)[0]
                
                batch_viewspace_point_grad = torch.stack(batch_point_grad,1).sum(1)
                batch_viewspace_point_grad[visibility_filter] = batch_viewspace_point_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)
                
                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone()[:,0].detach()
                    batch_t_grad[visibility_filter] = batch_t_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                    batch_t_grad = batch_t_grad.unsqueeze(1)
            else:
                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone().detach()
            
            iter_end.record()
            loss_dict = {"Ll1": Ll1,
                        "Lssim": Lssim}
            
            with torch.no_grad():
                #optimizer_start = time.time()
                psnr_for_log = psnr(image, gt_image).mean()
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item()# + 0.6 * ema_loss_for_log
                ema_l1loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_l1loss_for_log
                ema_ssimloss_for_log = 0.4 * Lssim.item() + 0.6 * ema_ssimloss_for_log
                
                for lambda_name in lambda_all:
                    if opt.__dict__[lambda_name] > 0:
                        ema = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                        vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"] = 0.4 * vars()[f"L{lambda_name.replace('lambda_', '')}"].item() + 0.6*ema
                        loss_dict[lambda_name.replace("lambda_", "L")] = vars()[lambda_name.replace("lambda_", "L")]
                        
                if iteration % 10 == 0:
                    postfix = {"Loss": f"{ema_loss_for_log:.{7}f}",
                                            "PSNR": f"{psnr_for_log:.{2}f}",
                                            "Ll1": f"{ema_l1loss_for_log:.{4}f}",
                                            "Lssim": f"{ema_ssimloss_for_log:.{4}f}",}
                    
                    for lambda_name in lambda_all:
                        if opt.__dict__[lambda_name] > 0:
                            ema_loss = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                            postfix[lambda_name.replace("lambda_", "L")] = f"{ema_loss:.{4}f}"
                            
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # # Log and save
                # test_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), loss_dict)
                # if (iteration in testing_iterations):
                #     if test_psnr >= best_psnr:
                #         best_psnr = test_psnr
                #         print("\n[ITER {}] Saving best checkpoint".format(iteration))
                #         torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt_best.pth")
                #         torch.save((tgh.capture(gaussians, opt), iteration), scene.model_path + "/tgh_chkpnt_best.pth")
                        
                if (iteration in saving_iterations):
                #if iteration % 100 == 0:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.tgh.clear_tgh()
                    scene.tgh.create_from_gaussians(gaussians)
                    scene.save(iteration, opt, tgh)

                # Densification
                if iteration < opt.densify_until_iter and (opt.densify_until_num_points < 0 or gaussians.get_xyz.shape[0] < opt.densify_until_num_points):
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    if batch_size == 1:
                        gaussians.add_densification_stats_pgsr(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter, batch_t_grad if gaussians.gaussian_dim == 4 else None)
                    else:
                        gaussians.add_densification_stats_grad(batch_viewspace_point_grad, visibility_filter, batch_t_grad if gaussians.gaussian_dim == 4 else None)
                        
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    #if iteration > 100:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.thresh_opa_prune, scene.cameras_extent, size_threshold, opt.densify_grad_t_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    #if iteration % 100 == 0:
                        gaussians.reset_opacity()
                        tgh.reset_opacity()

                if iteration == 500000:
                    tgh.reset_diffuse()
                    gaussians.reset_diffuse()
                        
                # Optimizer step
                if iteration < opt.iterations:
                    # copy_and_cat_engine.waitGroupCompletion(0, 0)
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    # if pipe.env_map_res and iteration < pipe.env_optimize_until:
                    #     env_map_optimizer.step()
                    #     env_map_optimizer.zero_grad(set_to_none = True)
                #optimizer_end = time.time()
                #torch.cuda.synchronize()
                #print(f"optimizer step time: {optimizer_end - optimizer_start:.6f} seconds")
                #cuda_to_cpu_start = time.time()
                if gaussian_init_flag:
                    #scene.tgh.update_from_gaussians(gaussians, opt, None)
                    pass
                else:
                    #scene.tgh.create_from_gaussians(gaussians)
                    gaussian_init_flag = True
                    state_dict = gaussians.get_state_dict()
                    gaussians_segments = []
                    for level in range(0, 10):
                        segment_count = len(tgh.layers[level])
                        for ind in range(segment_count):
                            gaussians_segments.append(tgh.layers[level][ind])
                            gaussians.append_from_gaussians_cpu(tgh.layers[level][ind])
                    #gaussians.clone_from_cpu(gaussians_segments)
                    gaussians.reset_param_groups()
                    gaussians.append_state_from_gaussians_cpu(gaussians_segments, state_dict)
                # if save_flag:
                    #torch.save((tgh.capture(gaussians), iteration), scene.model_path + "/tgh_chkpnt_best_after_prune.pth")
                #cuda_to_cpu_end = time.time()
                #torch.cuda.synchronize()
                #print(f"cuda to cpu time: {cuda_to_cpu_end - cuda_to_cpu_start:.6f} seconds")
            #scene.tgh.update_from_gaussians(gaussians, opt)
            #torch.cuda.synchronize()
            #end_t = time.time()
            #print(f"total iter time: {end_t - start_t:.6f} seconds")
            # torch.cuda.synchronize()
            # train_end = time.time()
            # print(f"train time:{train_end - train_start:.6f} seconds")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_dict=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        if loss_dict is not None:
            if "Lrigid" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/rigid_loss', loss_dict['Lrigid'].item(), iteration)
            if "Ldepth" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/depth_loss', loss_dict['Ldepth'].item(), iteration)
            if "Ltv" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/tv_loss', loss_dict['Ltv'].item(), iteration)
            if "Lopa" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/opa_loss', loss_dict['Lopa'].item(), iteration)
            if "Lptsopa" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/pts_opa_loss', loss_dict['Lptsopa'].item(), iteration)
            if "Lsmooth" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/smooth_loss', loss_dict['Lsmooth'].item(), iteration)
            if "Llaplacian" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/laplacian_loss', loss_dict['Llaplacian'].item(), iteration)

    psnr_test_iter = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        validation_configs = ({'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
                              {'name': 'test', 'cameras' : [scene.getTestCameras()[idx] for idx in range(len(scene.getTestCameras()))]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                msssim_test = 0.0
                for idx, batch_data in enumerate(tqdm(config['cameras'])):
                    gt_image, viewpoint = batch_data
                    gt_image = gt_image.cuda()
                    viewpoint = viewpoint.cuda()
                    
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    
                    depth = easy_cmap(render_pkg['depth'][0])
                    alpha = torch.clamp(render_pkg['alpha'], 0.0, 1.0).repeat(3,1,1)
                    if tb_writer and (idx < 5):
                        grid = [gt_image, image, alpha, depth]
                        grid = make_grid(grid, nrow=2)
                        tb_writer.add_images(config['name'] + "_view_{}/gt_vs_render".format(viewpoint.image_name), grid[None], global_step=iteration)
                            
                    l1_test += l1_loss(image, gt_image).mean()
                    psnr_test += psnr(image, gt_image).mean()
                    ssim_test += ssim(image, gt_image).mean()
                    msssim_test += msssim(image[None].cpu(), gt_image[None].cpu())
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 
                ssim_test /= len(config['cameras'])     
                msssim_test /= len(config['cameras'])        
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - msssim', msssim_test, iteration)
                if config['name'] == 'test':
                    psnr_test_iter = psnr_test.item()
                    
    torch.cuda.empty_cache()
    return psnr_test_iter

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_decay_weight(start_iteration, end_iteration, current_iteration):
    if current_iteration < start_iteration:
        return 1
    if current_iteration >= end_iteration:
        return 0.05
    return 1 * (end_iteration - current_iteration) / (end_iteration - start_iteration) + 0.1 * (current_iteration - start_iteration) / (end_iteration - start_iteration)

if __name__ == "__main__":
    # Set up command line argument parser
    #torch.multiprocessing.set_sharing_strategy('file_system')
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument('--num_pts', type=int, default=100_000)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    
    args = parser.parse_args(sys.argv[1:])
    #args.save_iterations.append(args.iterations)
        
    cfg = OmegaConf.load(args.config)
    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])
    for k in cfg.keys():
        recursive_merge(k, cfg)
        
    if args.exhaust_test:
        args.test_iterations = args.test_iterations + [i for i in range(0,args.iterations + 1,5000)]
    args.save_iterations = args.save_iterations + [i for i in range(2000,args.iterations + 1,2000)]
    setup_seed(args.seed)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.start_checkpoint, args.debug_from,
             args.gaussian_dim, args.time_duration, args.num_pts, args.num_pts_ratio, args.rot_4d, args.force_sh_3d, args.batch_size)

    # All done
    print("\nTraining complete.")
