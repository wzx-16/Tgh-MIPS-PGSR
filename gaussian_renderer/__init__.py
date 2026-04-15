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
from torch.nn import functional as F
import math
#from diff_gaussian_rasterization_4d_abs import GaussianRasterizationSettings, GaussianRasterizer
from diff_plane_rasterization_anti import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization_anti import GaussianRasterizer as PlaneGaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh, eval_shfs_4d
from utils.transformation_util import matrix_to_quaternion, quaternion_to_matrix
from utils.graphics_utils import focal2fov, getProjectionMatrix, normal_from_depth_image
from utils.color_utils import linear2srgb
from utils.sph_util import cart2sph
import numpy as np

def get_outside_msk(xyz, ENV_CENTER, ENV_RADIUS):
    if ENV_CENTER is None or ENV_RADIUS is None:
        #print("mask", (torch.zeros(xyz.shape[0], device="cuda", dtype=torch.bool)).size())
        return torch.zeros(xyz.shape[0], device="cuda", dtype=torch.bool)
    #print("mask", (torch.sum((xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2).size())
    return torch.sum((xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2

def reflection(rayd, normal):
    refl = rayd - 2*normal*torch.sum(rayd*normal, dim=-1, keepdim=True)
    return refl

def sample_cubemap_color(rays_d, env_map):
    H,W = rays_d.shape[:2]
    outcolor = torch.sigmoid(env_map(rays_d.reshape(-1,3)))
    outcolor = outcolor.reshape(H,W,3).permute(2,0,1)
    return outcolor

def sample_camera_rays(HWK, R, T, pixel_camera):
    pixel_camera = torch.tensor(pixel_camera).cuda()
    H,W,K = HWK
    R = R.T # NOTE!!! the R rot matrix is transposed save in 3DGS
    
    #global pixel_camera
    # if pixel_camera is None or pixel_camera.shape[0] != H:
    # K = K.astype(np.float32)
    # i, j = np.meshgrid(np.arange(W, dtype=np.float32),
    #                 np.arange(H, dtype=np.float32),
    #                 indexing='xy')
    # xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    # pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    # pixel_camera = torch.tensor(pixel_camera).cuda()

    rays_o = (-R.T @ T.unsqueeze(-1)).flatten()
    pixel_world = (pixel_camera - T[None, None]).reshape(-1, 3) @ R
    rays_d = pixel_world - rays_o[None]
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    rays_d = rays_d.reshape(H,W,3)
    #rays_d = rays_d.permute(1, 0, 2)
    return rays_d

def get_refl_dir(HWK, R, T, normal_map, pixel_camera): #RT W2C
    rays_d = sample_camera_rays(HWK, R, T, pixel_camera)
    #print("rays d", rays_d.shape)
    #print("normal map", normal_map.shape)
    rays_f = reflection(rays_d, normal_map)
    #rays_d = rays_d.clamp(-1, 1) # avoid numerical error when arccos
    return rays_f, rays_d

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc._rotation, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color if not pipe.env_map_res else torch.zeros(3, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        sh_degree_t=pc.active_sh_degree_t,
        campos=viewpoint_camera.camera_center,
        timestamp=viewpoint_camera.timestamp,
        time_duration=pc.time_duration[1]-pc.time_duration[0],
        rot_4d=pc.rot_4d,
        gaussian_dim=pc.gaussian_dim,
        force_sh_3d=pc.force_sh_3d,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    scales_t = None
    rotations = None
    rotations_r = None
    ts = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        if pc.rot_4d:
            cov3D_precomp, delta_mean = pc.get_current_covariance_and_mean_offset(scaling_modifier, viewpoint_camera.timestamp)
            means3D = means3D + delta_mean
        else:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        if pc.gaussian_dim == 4:
            marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)
            # marginal_t = torch.clamp_max(marginal_t, 1.0) # NOTE: 这里乘完会大于1，绝对不行——marginal_t应该用个概率而非概率密度 暂时可以clamp一下，后期用积分 —— 2d 也用的clamp
            opacity = opacity * marginal_t
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        if pc.gaussian_dim == 4:
            scales_t = pc.get_scaling_t
            ts = pc.get_t
            if pc.rot_4d:
                rotations_r = pc.get_rotation_r

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, pc.get_max_sh_channels)
            if pipe.compute_cov3D_python:
                dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)).detach()
            else:
                _, delta_mean = pc.get_current_covariance_and_mean_offset(scaling_modifier, viewpoint_camera.timestamp)
                dir_pp = ((means3D + delta_mean) - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)).detach()
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            if pc.gaussian_dim == 3 or pc.force_sh_3d:
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            elif pc.gaussian_dim == 4:
                dir_t = (pc.get_t - viewpoint_camera.timestamp).detach()
                sh2rgb = eval_shfs_4d(pc.active_sh_degree, pc.active_sh_degree_t, shs_view, dir_pp_normalized, dir_t, pc.time_duration[1] - pc.time_duration[0])
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
            if pc.gaussian_dim == 4 and ts is None:
                ts = pc.get_t
    else:
        colors_precomp = override_color
    
    flow_2d = torch.zeros_like(pc.get_xyz[:,:2])
    
    # Prefilter
    if pipe.compute_cov3D_python and pc.gaussian_dim == 4:
        mask = marginal_t[:,0] > 0.05
        if means2D is not None:
            means2D = means2D[mask]
        if means3D is not None:
            means3D = means3D[mask]
        if ts is not None:
            ts = ts[mask]
        if shs is not None:
            shs = shs[mask]
        if colors_precomp is not None:
            colors_precomp = colors_precomp[mask]
        if opacity is not None:
            opacity = opacity[mask]
        if scales is not None:
            scales = scales[mask]
        if scales_t is not None:
            scales_t = scales_t[mask]
        if rotations is not None:
            rotations = rotations[mask]
        if rotations_r is not None:
            rotations_r = rotations_r[mask]
        if cov3D_precomp is not None:
            cov3D_precomp = cov3D_precomp[mask]
        if flow_2d is not None:
            flow_2d = flow_2d[mask]
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, alpha, flow, covs_com = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        flow_2d = flow_2d,
        opacities = opacity,
        ts = ts,
        scales = scales,
        scales_t = scales_t,
        rotations = rotations,
        rotations_r = rotations_r,
        cov3D_precomp = cov3D_precomp)
    
    if pipe.env_map_res:
        assert pc.env_map is not None
        R = 60
        rays_o, rays_d = viewpoint_camera.get_rays()
        delta = ((rays_o*rays_d).sum(-1))**2 - (rays_d**2).sum(-1)*((rays_o**2).sum(-1)-R**2)
        assert (delta > 0).all()
        t_inter = -(rays_o*rays_d).sum(-1)+torch.sqrt(delta)/(rays_d**2).sum(-1)
        xyz_inter = rays_o + rays_d * t_inter.unsqueeze(-1)
        tu = torch.atan2(xyz_inter[...,1:2], xyz_inter[...,0:1]) / (2 * torch.pi) + 0.5 # theta
        tv = torch.acos(xyz_inter[...,2:3] / R) / torch.pi
        texcoord = torch.cat([tu, tv], dim=-1) * 2 - 1
        bg_color_from_envmap = F.grid_sample(pc.env_map[None], texcoord[None])[0] # 3,H,W
        # mask2 = (0 < xyz_inter[...,0]) & (xyz_inter[...,1] > 0) # & (xyz_inter[...,2] > -19)
        rendered_image = rendered_image + (1 - alpha) * bg_color_from_envmap # * mask2[None]
    
    if pipe.compute_cov3D_python and pc.gaussian_dim == 4:
        radii_all = radii.new_zeros(mask.shape)
        radii_all[mask] = radii
    else:
        radii_all = radii

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii_all > 0,
            "radii": radii_all,
            "depth": depth,
            "alpha": alpha,
            "flow": flow}

def render_3d_pgsr(
    viewpoint_camera,
    xyz: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    active_sh_degree,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    bg_color: torch.Tensor,
    scaling_modifier = 1.0,
    shs = None,
    mask = None,
    max_sh_channels=0,
):
    means3D = xyz
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(means3D, dtype = means3D.dtype, requires_grad = True, device = "cuda") + 0
    screenspace_points_abs = torch.zeros_like(means3D, dtype = means3D.dtype, requires_grad = True, device = "cuda") + 0
    
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass
    
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = opacities

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # cov3D_precomp = None
    # Set up rasterization configuration
    FoVx = viewpoint_camera.FoVx
    FoVy = viewpoint_camera.FoVy
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)
    # world_view_transform = extr.transpose(1, 0).cuda()
    # projection_matrix = getProjectionMatrix(znear = 0.1, zfar = 100, fovX = FoVx, fovY = FoVy, K = intr, img_w = img_w, img_h = img_h).transpose(0, 1).cuda()
    # full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    # camera_center = torch.linalg.inv(extr)[:3, 3]

    raster_settings = PlaneGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree = active_sh_degree,
            campos = viewpoint_camera.camera_center,
            prefiltered=False,
            render_geo=True,
            debug=False
        )

    rasterizer = PlaneGaussianRasterizer(raster_settings = raster_settings)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    assert not (shs is not None and colors is not None), "Cannot use both color and SH!"
    if colors is not None:
        colors_precomp = colors
    else:
        colors_precomp = None

    if mask is not None:
        means2D = means2D[mask]
        means2D_abs = means2D_abs[mask]
        means3D = means3D[mask]
        if colors_precomp is not None:
            colors_precomp = colors_precomp[mask]
        if shs is not None:
            shs = shs[mask]
        # cov3D_precomp = cov3D_precomp[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        opacity = opacity[mask]

    # cov = torch.cat([cov3D_precomp[..., :3], cov3D_precomp[..., 1:2], cov3D_precomp[..., 3:5], cov3D_precomp[..., 2:3], cov3D_precomp[..., 4:5], cov3D_precomp[..., 5:6]], dim=-1).reshape(-1, 3, 3)
    # # # print(cov)
    # eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    # # # scales = torch.ones_like(means3D[:, :3]) * 0.01
    # scales = eigenvalues[:, :]**0.5# + 1e-6
    # # print(scales.shape)
    # # print(scales.mean(), scales.std(), scales.min(), scales.max())
    # # mins, index = torch.min(eigenvalues[:, 0], dim=0, keepdim=True)
    # # print(mins, cov3D_precomp[index])
    # # # scales = scales * 0 + 0.01
    # # # scales = scales.detach()
    # # # scaling = gaussians_gpu.get_scaling
    # rotations = eigenvectors.detach()# + 1e-6
    # # rotations = eigenvectors#.detach()
    # # print(rotations.mean(), scales.mean())
    # # rotations = torch.zeros_like(means3D[:, [0]*4])
    # # rot = gaussians_gpu.get_rotation
    # # rotations[..., 0] = rotations[..., 0] + 1.0 # make sure the first quaternion component is always 1.0
    # # rotations = rotations.detach()
    # # print(cov, eigenvalues, eigenvectors)
    global_normal = get_normal(scales, rotations, viewpoint_camera.camera_center, means3D)
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
    pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3]
    depth_z = pts_in_cam[:, 2]
    local_distance = -(local_normal * pts_in_cam).sum(-1)
    input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
    input_all_map[:, :3] = local_normal
    input_all_map[:, 3] = 1.0
    input_all_map[:, 4] = local_distance

    # print(local_distance.min().data, local_distance.mean().data, local_distance.max().data, 'ddd')
    # print(torch.linalg.norm(input_all_map[:, :3], dim=-1).min(), torch.linalg.norm(input_all_map[:, :3], dim=-1).max(), 'ooo')

    rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        means2D_abs = means2D_abs,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        all_map = input_all_map,
        # cov3D_precomp = cov3D_precomp
        )

    # ray = means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1)
    # ray = ray / ray[..., 2:3]
    # print(input_all_map[(input_all_map[:, 4] / -(input_all_map[:, 0] * ray[:, 0] + input_all_map[:, 1] * ray[:, 1] + input_all_map[:, 2] + 1.0e-8))<-1100], 'ppp')

    if mask is not None:
        radii_all = radii.new_zeros(mask.shape)
        radii_all[mask] = radii
        radii = radii_all

    rendered_normal = out_all_map[0:3]
    rendered_alpha = out_all_map[3:4, ]
    rendered_distance = out_all_map[4:5, ]
    # print(rendered_distance.min(), rendered_distance.mean(), rendered_distance.max(), 'ddd')
    
    return_dict =  {"render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "viewspace_points_abs": screenspace_points_abs,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "out_observe": out_observe,
                    "rendered_normal": rendered_normal,
                    "depth": plane_depth,
                    "rendered_distance": rendered_distance,
                    'alpha': rendered_alpha,
                    # 'index': index,
                    # 'scales': scales,
                    }

    # depth = plane_depth
    # u_map = torch.ones((viewpoint_camera.image_height, 1), device=depth.device) * torch.arange(1, viewpoint_camera.image_width + 1, device=depth.device) - viewpoint_camera.cx  # u-u0
    # v_map = torch.arange(1, viewpoint_camera.image_height + 1, device=depth.device).reshape(viewpoint_camera.image_height, 1) * torch.ones((1, viewpoint_camera.image_width), device=depth.device) - viewpoint_camera.cy  # v-v0

    # VERSION = 'd2nt_v3'
    # # get depth gradients
    # if VERSION == 'd2nt_basic':
    #     Gu, Gv = get_filter(depth[None])
    # else:
    #     Gu, Gv = get_DAG_filter(depth[None])

    # # Depth to Normal Translation
    # est_nx = Gu[0, 0] * viewpoint_camera.fl_x
    # est_ny = Gv[0, 0] * viewpoint_camera.fl_y
    # est_nz = -(depth[0] + v_map * Gv[0, 0] + u_map * Gu[0, 0])
    # est_normal = torch.stack((est_nx, est_ny, est_nz), dim=0)
    # # print(est_nx.shape)
    # # exit()
    # # vector normalization
    
    # est_normal = torch.nn.functional.normalize(est_normal[None], dim=1)

    # # MRF-based Normal Refinement
    # if VERSION == 'd2nt_v3':
    #     est_normal = MRF_optim(depth[None], est_normal)

    # depth_normal = est_normal[0]
    # # print(depth_normal.mean())
    depth_normal = render_normal(viewpoint_camera.intr, torch.linalg.inv(viewpoint_camera.extr), plane_depth.squeeze()) * (rendered_alpha).detach()
    return_dict.update({"depth_normal": depth_normal})
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return return_dict

def render_3d_pgsr_anti(
    viewpoint_camera,
    xyz: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    active_sh_degree,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    bg_color: torch.Tensor,
    scaling_modifier = 1.0,
    shs = None,
    mask = None,
    max_sh_channels=0,
    normal = None,
    reflect = None,
    dir_pp = None,
    pc: GaussianModel = None,
    iteration = 0,
    timestamp = None,

):
    means3D = xyz
    if timestamp is None:
        timestamp = viewpoint_camera.timestamp
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(means3D, dtype = means3D.dtype, requires_grad = True, device = "cuda") + 0
    screenspace_points_abs = torch.zeros_like(means3D, dtype = means3D.dtype, requires_grad = True, device = "cuda") + 0
    
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass
    
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = opacities

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # cov3D_precomp = None
    # Set up rasterization configuration
    FoVx = viewpoint_camera.FoVx
    FoVy = viewpoint_camera.FoVy
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)
    # world_view_transform = extr.transpose(1, 0).cuda()
    # projection_matrix = getProjectionMatrix(znear = 0.1, zfar = 100, fovX = FoVx, fovY = FoVy, K = intr, img_w = img_w, img_h = img_h).transpose(0, 1).cuda()
    # full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    # camera_center = torch.linalg.inv(extr)[:3, 3]
    #print(viewpoint_camera.extr, "test")
    raster_settings = PlaneGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            kernel_size=0.3,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree = active_sh_degree,
            campos = viewpoint_camera.camera_center,
            prefiltered=False,
            render_geo=True,
            debug=False
        )
    #print(viewpoint_camera.extr, "1")
    rasterizer = PlaneGaussianRasterizer(raster_settings = raster_settings)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    assert not (shs is not None and colors is not None), "Cannot use both color and SH!"
    if colors is not None:
        colors_precomp = colors
    else:
        colors_precomp = None

    xyz = pc.get_xyz + pc.get_velocity * (viewpoint_camera.timestamp - pc.get_t) / (pc.get_sigma_t + 1)
    view_pos = viewpoint_camera.camera_center
    diffuse = pc.get_diffuse(dir_pp)
    rgb = eval_sh(pc.active_sh_degree, pc.get_features.transpose(1, 2), dir_pp)
    rgb = torch.clamp_min(rgb + 0.5, 0.0)
    ENV_CENTER = torch.tensor([0, 0, 0], device="cuda")
    ENV_RADIUS = 8
    outside_mask = get_outside_msk(means3D, ENV_CENTER, ENV_RADIUS)
    gs_in = torch.ones_like(opacity)
    gs_in[outside_mask] = 0.0
    gs_out = 1.0 - gs_in.detach()
    #diffuse = torch.clamp_min(diffuse + 0.5, 0.0)
    #diffuse   = torch.logit(pc.get_diffuse.clamp(0 + 1e-6, 1 - 1e-6)) + 1.098612
    specular  = pc.get_specular
    albedo = pc.get_albedo
    specular2 = pc.get_specular2
    #specular2 = pc._specular2
    roughness = pc.get_roughness
    color = torch.zeros_like(diffuse)
    # if iteration > 1000000:
    #     color = pc.brdf_mlp.shade(xyz[None, None, ...].detach(), normal[None, None, ...], reflect[None, None, ...], diffuse[None, None, ...], specular[None, None, ...], roughness[None, None, ...], view_pos[None, None, ...], iteration)
    #     #color2 = pc.brdf_mlp_2.shade_without_diffuse(xyz[None, None, ...].detach(), normal[None, None, ...], reflect[None, None, ...], diffuse[None, None, ...], specular2[None, None, ...], roughness[None, None, ...], view_pos[None, None, ...], iteration)
    #     #color = color + color2
    #     #color = diffuse
    #     #color = linear2srgb(color)
    #     shs = None
    #     colors_precomp = color.squeeze() 
    # elif iteration < 0:
    #     #color = torch.sigmoid(diffuse - np.log(3.0))
    #     color = diffuse
    #     colors_precomp = color.squeeze() 
    #     shs = None
    if iteration < 15000:
        color = rgb
        colors_precomp = color.squeeze() 
        shs = None
    else:
        color[~outside_mask] = diffuse[~outside_mask]
        color[outside_mask] = rgb[outside_mask]
        colors_precomp = color.squeeze() 
        shs = None

    if mask is not None:
        means2D = means2D[mask]
        means2D_abs = means2D_abs[mask]
        means3D = means3D[mask]
        if colors_precomp is not None:
            colors_precomp = colors_precomp[mask]
        if shs is not None:
            shs = shs[mask]
        # cov3D_precomp = cov3D_precomp[mask]
        xyz = xyz[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        opacity = opacity[mask]
        specular = specular[mask]
        roughness = roughness[mask]
        specular2 = specular2[mask]
        gs_in = gs_in[mask]
        gs_out = gs_out[mask]
        diffuse = diffuse[mask]
        albedo = albedo[mask]

    #shs = None

    # cov = torch.cat([cov3D_precomp[..., :3], cov3D_precomp[..., 1:2], cov3D_precomp[..., 3:5], cov3D_precomp[..., 2:3], cov3D_precomp[..., 4:5], cov3D_precomp[..., 5:6]], dim=-1).reshape(-1, 3, 3)
    # # # print(cov)
    # eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    # # # scales = torch.ones_like(means3D[:, :3]) * 0.01
    # scales = eigenvalues[:, :]**0.5# + 1e-6
    # # print(scales.shape)
    # # print(scales.mean(), scales.std(), scales.min(), scales.max())
    # # mins, index = torch.min(eigenvalues[:, 0], dim=0, keepdim=True)
    # # print(mins, cov3D_precomp[index])
    # # # scales = scales * 0 + 0.01
    # # # scales = scales.detach()
    # # # scaling = gaussians_gpu.get_scaling
    # rotations = eigenvectors.detach()# + 1e-6
    # # rotations = eigenvectors#.detach()
    # # print(rotations.mean(), scales.mean())
    # # rotations = torch.zeros_like(means3D[:, [0]*4])
    # # rot = gaussians_gpu.get_rotation
    # # rotations[..., 0] = rotations[..., 0] + 1.0 # make sure the first quaternion component is always 1.0
    # # rotations = rotations.detach()
    # # print(cov, eigenvalues, eigenvectors)
    global_normal = get_normal(scales, rotations, viewpoint_camera.camera_center, means3D)
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
    pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3]
    depth_z = pts_in_cam[:, 2]
    local_distance = -(local_normal * pts_in_cam).sum(-1)
    input_all_map = torch.zeros((means3D.shape[0], 32)).cuda().float()
    input_all_map[:, :3] = local_normal
    input_all_map[:, 3] = 1.0
    input_all_map[:, 4] = local_distance
    #print("specular", specular)
    #input_all_map[:, 5:8] = specular
    input_all_map[:, 5:8] = specular
    input_all_map[:, 8:9] = roughness
    input_all_map[:, 9:12] = global_normal
    cos_feature = torch.tensor([np.cos(np.pi * (2**i) * 4 * timestamp) for i in range(5)], dtype=torch.float32, device='cuda')
    #cos_feature = torch.tensor([np.cos(np.pi * i * 10 * timestamp) for i in range(4)], dtype=torch.float32, device='cuda')
    sin_feature = torch.tensor([np.sin(np.pi * (2**i) * 4 * timestamp) for i in range(5)], dtype=torch.float32, device='cuda')
    fourier_feature = torch.cat([cos_feature, sin_feature], dim=0)
    feature = specular2[:, : pc.gsdim]
    # timestamp_feature = torch.tensor([timestamp], dtype=torch.float32, device='cuda').repeat(feature.shape[0], 1)
    # light_mlp2_input = torch.cat([torch.tanh(feature), xyz, timestamp_feature], dim=-1)
    # delta_feature = pc.light_mlp_2(light_mlp2_input)
    # feature = torch.tanh(feature + delta_feature)
    feature_coeff = specular2[:, pc.gsdim :] * 2
    if iteration >= 35000:
        feature = feature + (feature_coeff.reshape(-1, pc.gsdim, 10) @ fourier_feature).squeeze()
    #feature = feature + (feature_coeff.reshape(-1, pc.gsdim, 4) @ cos_feature).squeeze()
    input_all_map[:, 12:16] = feature
    input_all_map[:, 16:19] = albedo
    input_all_map[:, 27:28] = gs_in
    input_all_map[:, 28:29] = gs_out
    input_all_map[:, 29:32] = diffuse
    #input_all_map[:, 12:16] = specular2

    # print(local_distance.min().data, local_distance.mean().data, local_distance.max().data, 'ddd')
    # print(torch.linalg.norm(input_all_map[:, :3], dim=-1).min(), torch.linalg.norm(input_all_map[:, :3], dim=-1).max(), 'ooo')
    # torch.cuda.synchronize()
    # print(viewpoint_camera.extr, "2")
    # extr_before = viewpoint_camera.extr.clone()
    rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        means2D_abs = means2D_abs,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        all_map = input_all_map,
        # cov3D_precomp = cov3D_precomp
        )
    # torch.cuda.synchronize()
    # print(viewpoint_camera.extr, "3")
    # print("extr id:", id(viewpoint_camera.extr))
    # print("Has extr changed?", not torch.allclose(viewpoint_camera.extr, extr_before))
    # print("extr after:\n", viewpoint_camera.extr)

    # ray = means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1)
    # ray = ray / ray[..., 2:3]
    # print(input_all_map[(input_all_map[:, 4] / -(input_all_map[:, 0] * ray[:, 0] + input_all_map[:, 1] * ray[:, 1] + input_all_map[:, 2] + 1.0e-8))<-1100], 'ppp')

    if mask is not None:
        # print(mask.shape)
        # print(radii.shape)
        #torch.cuda.synchronize()

        radii_all = radii.new_zeros(mask.shape)
        radii_all[mask] = radii
        radii = radii_all

    rendered_normal = out_all_map[0:3]
    rendered_alpha = out_all_map[3:4, ]
    rendered_distance = out_all_map[4:5, ]
    rendered_specular = out_all_map[5:8]
    rendered_in = out_all_map[27:28]
    rendered_out = out_all_map[28:29]
    rendered_roughness = out_all_map[8:9]
    rendered_global_normal = out_all_map[9:12]
    rendered_specular2 = out_all_map[12:16]
    rendered_albedo = out_all_map[16:19]
    rendered_diffuse = out_all_map[29:32]
    rendered_global_normal = torch.nn.functional.normalize(rendered_global_normal.permute(1, 2, 0), dim=2, eps=1e-6)
    rendered_specular = rendered_specular.permute(1, 2, 0)
    rendered_in = rendered_in.permute(1, 2, 0)
    rendered_out = rendered_out.permute(1, 2, 0)
    rendered_roughness = rendered_roughness.permute(1, 2, 0)
    rendered_specular2 = rendered_specular2.permute(1, 2, 0)
    rendered_diffuse = rendered_diffuse.permute(1, 2, 0)
    feature_map = rendered_specular2
    # feature_map = rendered_specular2.reshape(-1, 15)[:, : 4]
    # feature_time = rendered_specular2.reshape(-1, 15)[:, 4:8]
    # feature_map = rendered_specular2.reshape(-1, 15)[:, : 8]
    # feature_time = rendered_specular2.reshape(-1, 15)[:, 8:12]
    # fourier_coeff = rendered_specular2.reshape(-1, 84)[:, pc.gsdim :] * 2
    # cos_feature = torch.tensor([np.cos(np.pi * i * timestamp) for i in range(10)], dtype=torch.float32, device='cuda')
    # sin_feature = torch.tensor([np.sin(np.pi * i * timestamp) for i in range(10)], dtype=torch.float32, device='cuda')
    # fourier_feature = torch.cat([cos_feature, sin_feature], dim=0)
    # #feature_map = feature_map * fourier_feature[None, :]
    # feature_map = feature_map + (fourier_coeff.reshape(-1, pc.gsdim, 20) @ fourier_feature).squeeze()
    feature_map = F.normalize(feature_map, dim=-1)
    feature_map = feature_map.reshape(-1, 1, 4)
    #feature_map = feature_map.reshape(-1, 1, 8)
    #print("feature map", feature_map)
    spec_coeff = None
    spec_rgb = None
    with torch.no_grad():
        select_index = (rendered_in.reshape(-1,) > 0.05).nonzero(as_tuple=True)[0]
        #select_index = torch.ones_like(rendered_in.reshape(-1,), dtype=torch.bool)[0]
    if iteration >= 15000:
        K = np.zeros((3,3))
        K[0][0] = viewpoint_camera.fl_x
        K[0][2] = viewpoint_camera.cx
        K[1][1] = viewpoint_camera.fl_y
        K[1][2] = viewpoint_camera.cy
        K[2][2] = 1
        HWK = (viewpoint_camera.H, viewpoint_camera.W, K)
        R = torch.tensor(viewpoint_camera.R, dtype=torch.float32, device='cuda')
        T = torch.tensor(viewpoint_camera.T, dtype=torch.float32, device='cuda')
        #print(rendered_global_normal.size())
        reflec_dir, rays_d = get_refl_dir(HWK, R, T, rendered_global_normal, viewpoint_camera.pixel_camera)
        reflec_dir_selected = reflec_dir.reshape(-1, 3)[select_index]
        wo = -rays_d
        wo_xy = (cart2sph(reflec_dir_selected.reshape(-1, 3)[..., [2,1,0]])[..., 1:] / torch.Tensor([[np.pi, 2*np.pi]]).cuda())[..., [1,0]] 

        wo_xyz = torch.stack([wo_xy[:, None, :]], dim=0,)
        #print(wo_xyz)
        spec_level = rendered_roughness.reshape(-1, 1)[select_index]

        spec_feat = pc.dir_encoding(wo_xyz, spec_level.view(-1, 1), index=0, timestamp=timestamp).reshape(-1, pc.sph_dim)
        #reflectance = pc.brdf_mlp.shade_fg(wo.unsqueeze(2), rendered_global_normal.unsqueeze(2), rendered_specular.unsqueeze(2), rendered_roughness.unsqueeze(2))
        #spec_feat = spec_feat * reflectance.reshape(-1, 3)[select_index].mean(-1, keepdim=True)
        #print(spec_feat)
        spec_feat_wrap = spec_feat.reshape(-1, pc.sph_dim, 1)
        spec_feat_dirc = spec_feat.reshape(-1, pc.sph_dim)
        wrap_input = (spec_feat_wrap @ feature_map[select_index]).reshape(-1, pc.sph_dim* pc.gsdim)
        #input_mlp = torch.cat([wrap_input, spec_feat_dirc, reflectance.reshape(-1, 3)[select_index]], -1)
        # cos_feature = torch.tensor([np.cos(np.pi * (2**i) * 2 * timestamp) for i in range(1, 6)], dtype=torch.float32, device='cuda')
        # sin_feature = torch.tensor([np.sin(np.pi * (2**i) * 2 * timestamp) for i in range(1, 6)], dtype=torch.float32, device='cuda')
        # fourier_feature = torch.cat([cos_feature, sin_feature], dim=0)
        # input_mlp2 = torch.cat([feature_map[select_index].reshape(-1, 4), feature_time[select_index].reshape(-1, 4), fourier_feature.unsqueeze(0).expand(feature_map[select_index].shape[0], -1)], dim=-1)
        # feature_output = F.normalize(pc.light_mlp_2(input_mlp2).float(), dim=-1)
        #wrap_input = (spec_feat_wrap @ feature_output.reshape(-1, 1, 4)).reshape(-1, pc.sph_dim * 4)
        #input_mlp = torch.cat([wrap_input, spec_feat_dirc, fourier_feature.unsqueeze(0).expand(wrap_input.shape[0], -1)], -1)
        #wrap_input_feature = (fourier_feature.unsqueeze(0).expand(spec_feat_wrap.shape[0], -1).reshape(-1, 10, 1) @ feature_map[select_index]).reshape(-1, pc.gsdim * 10)
        # wrap_input_feature_time = (fourier_feature.unsqueeze(0).expand(spec_feat_wrap.shape[0], -1).reshape(-1, 10, 1) @ feature_time[select_index].reshape(-1, 1, 4)).reshape(-1, 4 * 10)
        # wrap_input_feature = torch.cat([feature_map[select_index].reshape(-1, 4), wrap_input_feature_time], dim=-1)
        # wrap_input = (spec_feat_wrap @ wrap_input_feature.reshape(-1, 1, 44)).reshape(-1, pc.sph_dim * 44)
        # wrap_input_feature_time = (fourier_feature.unsqueeze(0).expand(spec_feat_wrap.shape[0], -1).reshape(-1, 10, 1) @ feature_time[select_index].reshape(-1, 1, 4)).reshape(-1, 4 * 10)
        # wrap_input_feature = torch.cat([feature_map[select_index].reshape(-1, 8), wrap_input_feature_time], dim=-1)
        # wrap_input = (spec_feat_wrap @ wrap_input_feature.reshape(-1, 1, 48)).reshape(-1, pc.sph_dim * 48)
        # wrap_input_feature_time = (fourier_feature.unsqueeze(0).expand(spec_feat_wrap.shape[0], -1).reshape(-1, 10, 1) @ feature_time[select_index].reshape(-1, 1, 11)).reshape(-1, 11 * 10)
        # wrap_input_feature = torch.cat([feature_map[select_index].reshape(-1, 4), wrap_input_feature_time], dim=-1)
        # wrap_input = (spec_feat_wrap @ wrap_input_feature.reshape(-1, 1, 114)).reshape(-1, pc.sph_dim * 114)
        #wrap_input = (spec_feat_wrap @ wrap_input_feature.reshape(-1, 1, 80)).reshape(-1, pc.sph_dim * 80)
        input_mlp = torch.cat([wrap_input, spec_feat_dirc], -1)
        #print(input_mlp)
        mlp_output = pc.light_mlp(input_mlp).float()

        spec_light = torch.exp(torch.clamp(mlp_output, max=5.0))
        # rendered_albedo = rendered_specular.reshape(-1, 3)[select_index]
        # spec_rgb = linear2srgb(rendered_albedo + spec_light)
        # #spec_rgb = linear2srgb(spec_light)
        # spec_rgb = torch.clamp(spec_rgb, 0.0, 1.0)
        # output_rgb = torch.zeros(viewpoint_camera.H, viewpoint_camera.W, 3).cuda()
        # output_rgb.reshape(-1, 3)[select_index] = spec_rgb
        # output_rgb = output_rgb.permute(2, 0, 1)
        #spec_light = torch.exp(torch.tanh(mlp_output / 2) * 5.0)
        #spec_light = torch.exp(torch.where(mlp_output > 0, 5.0 * torch.tanh(mlp_output / 2.0), mlp_output))
        #spec_light = spec_light * reflectance.reshape(-1, 3)[select_index]
        #spec_light = torch.sign(mlp_output) * (torch.exp(torch.tanh(mlp_output) * 5.0) - 1)


        spec_rgb = torch.zeros(viewpoint_camera.H, viewpoint_camera.W, 3).cuda()
        spec_rgb.reshape(-1, 3)[select_index] = spec_light
        spec_rgb = spec_rgb.permute(2, 0, 1)


        # spec_coeff = torch.zeros((viewpoint_camera.H, viewpoint_camera.W, 1), device="cuda")
        # spec_coeff.reshape(-1, 1)[select_index] = torch.sigmoid(mlp_output)
        # #print("spec coeff", spec_coeff.size())
        # spec_coeff = spec_coeff.permute(2, 0, 1)

        #print(spec_coeff)
        # print("wo", wo.shape, wo.min().item(), wo.max().item())
        # print("refl", reflec_dir.shape, reflec_dir.min().item(), reflec_dir.max().item())
        # print("nrm len", rendered_global_normal.norm(dim=-1).mean().item())
        #specular_image = pc.brdf_mlp.shade_without_diffuse(wo.unsqueeze(2), reflec_dir.unsqueeze(2), rendered_global_normal.unsqueeze(2), rendered_specular.unsqueeze(2), rendered_roughness.unsqueeze(2), iteration)
        # if iteration > 10:
        #rendered_image = rendered_image + specular_image.squeeze().permute(2, 0, 1)
        #print("spec light", spec_light.size())
        #rendered_image = linear2srgb(rendered_image + specular_image.squeeze().permute(2, 0, 1) * spec_coeff)
        #rendered_image = linear2srgb(rendered_image + spec_rgb * rendered_specular.permute(2, 0, 1))
        #rendered_image = rendered_image * (rendered_out.permute(2, 0, 1)) + output_rgb * rendered_in.permute(2, 0, 1)
        rendered_image = linear2srgb(rendered_image + spec_rgb)
    else:
        #rendered_image = rendered_image
        rendered_image = linear2srgb(rendered_image)
    # print(rendered_distance.min(), rendered_distance.mean(), rendered_distance.max(), 'ddd')
    
    return_dict =  {"render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "viewspace_points_abs": screenspace_points_abs,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "out_observe": out_observe,
                    "rendered_normal": rendered_normal,
                    "depth": plane_depth,
                    "rendered_distance": rendered_distance,
                    'alpha': rendered_alpha,
                    "rendered_spec": rendered_specular,
                    "rendered_rough": rendered_roughness,
                    "rendered_gb_normal": rendered_global_normal,
                    "rendered_feature": rendered_specular2,
                    "feature_map": feature_map.reshape(viewpoint_camera.H, viewpoint_camera.W, 4).permute(2,0,1),
                    "spec_coeff": spec_coeff,
                    "spec_rgb": spec_rgb if spec_rgb is not None else None,
                    "rendered_in": rendered_in.permute(2, 0, 1),
                    # 'index': index,
                    # 'scales': scales,
                    }

    # depth = plane_depth
    # u_map = torch.ones((viewpoint_camera.image_height, 1), device=depth.device) * torch.arange(1, viewpoint_camera.image_width + 1, device=depth.device) - viewpoint_camera.cx  # u-u0
    # v_map = torch.arange(1, viewpoint_camera.image_height + 1, device=depth.device).reshape(viewpoint_camera.image_height, 1) * torch.ones((1, viewpoint_camera.image_width), device=depth.device) - viewpoint_camera.cy  # v-v0

    # VERSION = 'd2nt_v3'
    # # get depth gradients
    # if VERSION == 'd2nt_basic':
    #     Gu, Gv = get_filter(depth[None])
    # else:
    #     Gu, Gv = get_DAG_filter(depth[None])

    # # Depth to Normal Translation
    # est_nx = Gu[0, 0] * viewpoint_camera.fl_x
    # est_ny = Gv[0, 0] * viewpoint_camera.fl_y
    # est_nz = -(depth[0] + v_map * Gv[0, 0] + u_map * Gu[0, 0])
    # est_normal = torch.stack((est_nx, est_ny, est_nz), dim=0)
    # # print(est_nx.shape)
    # # exit()
    # # vector normalization
    
    # est_normal = torch.nn.functional.normalize(est_normal[None], dim=1)

    # # MRF-based Normal Refinement
    # if VERSION == 'd2nt_v3':
    #     est_normal = MRF_optim(depth[None], est_normal)

    # depth_normal = est_normal[0]
    # # print(depth_normal.mean())
    #print(viewpoint_camera.extr)
    depth_normal = render_normal(viewpoint_camera.intr, torch.linalg.inv(viewpoint_camera.extr), plane_depth.squeeze()) * (rendered_alpha).detach()
    return_dict.update({"depth_normal": depth_normal})
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return return_dict

# def get_smallest_axis(scaling, rotation, return_idx=False):
#     rotation_matrices = rotation
#     smallest_axis_idx = scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
#     smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
#     if return_idx:
#         return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
#     return smallest_axis.squeeze(dim=2)

def get_smallest_axis(scaling, rotation, return_idx=False):
    rotation_matrices = get_rotation_matrix(rotation)
    smallest_axis_idx = scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
    smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
    if return_idx:
        return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
    return smallest_axis.squeeze(dim=2)

def get_normal(scaling, rotation, camera_center, xyz):
    normal_global = get_smallest_axis(scaling, rotation)
    # normal_global = rotation
    gaussian_to_cam_global = camera_center - xyz
    neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
    normal_global[neg_mask] = -normal_global[neg_mask]
    return normal_global

def get_rotation_matrix(rotation):
    return quaternion_to_matrix(rotation)

def render_normal(intrinsic_matrix, extrinsic_matrix, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
                                            intrinsic_matrix.to(depth.device), 
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref