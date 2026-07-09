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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from tqdm import tqdm
import torch
from utils.general_utils import fps
from multiprocessing.pool import ThreadPool
import imagesize
import glob

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    timestamp: float = 0.0
    fl_x: float = -1.0
    fl_y: float = -1.0
    cx: float = -1.0
    cy: float = -1.0



def _parse_frame_filter(frame_filter):
    if frame_filter is None or frame_filter == "":
        return None
    if isinstance(frame_filter, (list, tuple, set)):
        return {int(value) for value in frame_filter}
    frames = set()
    for part in str(frame_filter).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            frames.update(range(int(start), int(end) + 1))
        else:
            frames.add(int(part))
    return frames


def _frame_from_image_name(image_name):
    token = Path(str(image_name)).stem.rsplit("_", 1)[-1]
    try:
        return int(token)
    except ValueError:
        return None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        timestamp = 0.0
        try:
            timestamp = int(image_name.split("_")[-1]) / 30.0
        except:
            pass
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, timestamp=timestamp)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    if 'nx' in vertices:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros_like(positions)
    if 'time' in vertices:
        timestamp = vertices['time'][:, None]
    else:
        timestamp = None
    return BasicPointCloud(points=positions, colors=colors, normals=normals, time=timestamp)

def fetchPlyList(paths):
    positions = None
    colors = None
    normals = None
    for path in paths:
        plydata = PlyData.read(path)
        vertices = plydata['vertex']
        if positions is None:
            positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
            colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
            if 'nx' in vertices:
                normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
            else:
                normals = np.zeros_like(positions)
            if 'time' in vertices:
                timestamp = vertices['time'][:, None]
            else:
                frame_num = int(path.split("/")[-1].split("_")[-1].split(".")[0], base=10)
                timestamp = np.full((vertices['x'].shape[0], 1), frame_num * (1/30), dtype=float)
        else:
            positions = np.vstack([positions, np.vstack([vertices['x'], vertices['y'], vertices['z']]).T])
            colors = np.vstack([colors, np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0])
            if 'nx' in vertices:
                normals = np.vstack([normals, np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T])
            else:
                normals = np.vstack([normals, np.zeros_like(positions)])
            if 'time' in vertices:
                timestamp = np.vstack([timestamp, vertices['time'][:, None]])
            else:
                frame_num = int(path.split("/")[-1].split("_")[-1].split(".")[0], base=10)
                timestamp = np.vstack([timestamp, np.full((vertices['x'].shape[0], 1), frame_num * (1/30), dtype=float)])
    return BasicPointCloud(points=positions, colors=colors, normals=normals, time=timestamp)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, num_pts_ratio=1.0):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    if num_pts_ratio > 1.001:
        num_pts = int((num_pts_ratio - 1) * pcd.points.shape[0])
        mean_xyz = pcd.points.mean(axis=0)
        min_rand_xyz = mean_xyz - np.array([0.5, 0.5, 0.5])
        max_rand_xyz = mean_xyz + np.array([0.5, 2.0, 0.5])
        xyz = np.concatenate([pcd.points, 
                              np.random.random((num_pts, 3)) * (max_rand_xyz - min_rand_xyz) + min_rand_xyz], 
                              axis=0)
        colors = np.concatenate([pcd.colors, 
                              SH2RGB(np.random.random((num_pts, 3)) / 255.0)], 
                              axis=0)
        normals = np.concatenate([pcd.normals, 
                              np.zeros((num_pts, 3))], 
                              axis=0)
        pcd = BasicPointCloud(points=xyz, colors=colors, normals=normals)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", time_duration=None, frame_ratio=1, dataloader=False, frame_filter=None):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
    if "camera_angle_x" in contents:
        fovx = contents["camera_angle_x"]
        
    frames = contents["frames"]
    frame_filter_set = _parse_frame_filter(frame_filter)
    tbar = tqdm(range(len(frames)))
    def frame_read_fn(idx_frame):
        idx = idx_frame[0]
        frame = idx_frame[1]
        cam_name = os.path.join(path, frame["file_path"] + extension)
        image_name = Path(cam_name).stem
        if frame_filter_set is not None:
            frame_id = _frame_from_image_name(image_name)
            if frame_id not in frame_filter_set:
                return

        timestamp = frame.get('time', 0.0)
        if frame_ratio > 1:
            timestamp /= frame_ratio
        if time_duration is not None and 'time' in frame:
            if timestamp < time_duration[0] or timestamp > time_duration[1]:
                return

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = os.path.join(path, cam_name) # .replace('hdImgs_unditorted', 'hdImgs_unditorted_rgba').replace('.jpg', '.png')
        #loaded_mask = None
        if not dataloader:
            with Image.open(image_path) as image_load:
                im_data = np.array(image_load.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            if norm_data[:, :, 3:4].min() < 1:
                arr = np.concatenate([arr, norm_data[:, :, 3:4]], axis=2)
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")
            else:
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            #width, height = image.size[0], image.size[1]
            width = contents["w"]
            height = contents["h"]
            # mask_path = "/" + os.path.join(os.path.join(*image_path.split("/")[0:-2]), os.path.join("mattings", image_name.split("_")[0]))
            # mask_name = image_name.split("_")[-1].split(".")[0] + ".png"
            # with Image.open(os.path.join(mask_path, mask_name)) as image_load:
            #     loaded_mask_PIL = image_load.resize((width, height))
            # loaded_mask = np.array(loaded_mask_PIL)
        else:
            image = np.empty(0)
            #width, height = imagesize.get(image_path)
            width = contents["w"]
            height = contents["h"]
        
        if 'depth_path' in frame:
            depth_name = frame["depth_path"]
            if not extension in frame["depth_path"]:
                depth_name = frame["depth_path"] + extension
            depth_path = os.path.join(path, depth_name)
            depth = Image.open(depth_path).copy()
        else:
            depth = None
        tbar.update(1)
        if 'fl_x' in frame and 'fl_y' in frame and 'cx' in frame and 'cy' in frame:
            #FovX = FovY = -1.0
            fl_x = frame['fl_x']
            fl_y = frame['fl_y']
            cx = frame['cx']
            cy = frame['cy']
            FovX = focal2fov(fl_x, width)
            FovY = focal2fov(fl_y, height)
            return CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth,
                        image_path=image_path, image_name=image_name, width=width, height=height, timestamp=timestamp,
                        fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy)
            
        elif 'fl_x' in contents and 'fl_y' in contents and 'cx' in contents and 'cy' in contents:
            FovX = FovY = -1.0
            fl_x = contents['fl_x']
            fl_y = contents['fl_y']
            cx = contents['cx']
            cy = contents['cy']
            return CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth,
                        image_path=image_path, image_name=image_name, width=width, height=height, timestamp=timestamp,
                        fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy)
        else:
            fovy = focal2fov(fov2focal(fovx, width), height)
            FovY = fovy
            FovX = fovx
            return CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth,
                            image_path=image_path, image_name=image_name, width=width, height=height, timestamp=timestamp)
    
    with ThreadPool() as pool:
        cam_infos = pool.map(frame_read_fn, zip(list(range(len(frames))), frames))
        pool.close()
        pool.join()
        
    cam_infos = [cam_info for cam_info in cam_infos if cam_info is not None]
    
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", num_pts=100_000, time_duration=None, num_extra_pts=0, frame_ratio=1, dataloader=False, render=False, frame_filter=None):
    
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, time_duration=time_duration, frame_ratio=frame_ratio, dataloader=dataloader, frame_filter=frame_filter)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json" if not path.endswith('lego') else "transforms_val.json", white_background, extension, time_duration=time_duration, frame_ratio=frame_ratio, dataloader=dataloader, frame_filter=frame_filter)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if not render:
        file_format = "points3d_*.ply"
        ply_paths = glob.glob(f"{path}/{file_format}")
        ply_path_origin = os.path.join(path, "points3d.ply")
        if len(ply_paths) == 0:
            if not os.path.exists(ply_path_origin):
                # Since this data set has no colmap data, we start with random points
                print(f"Generating random point cloud ({num_pts})...")
                
                # We create random points inside the bounds of the synthetic Blender scenes
                xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
                shs = np.random.random((num_pts, 3)) / 255.0
                pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

                storePly(ply_path_origin, xyz, SH2RGB(shs) * 255)
            try:
                pcd = fetchPly(ply_path_origin)
            except:
                pcd = None
        else:
            pcd = fetchPlyList(ply_paths)

        if pcd.points.shape[0] > num_pts:
            mask = np.random.randint(0, pcd.points.shape[0], num_pts)
            # mask = fps(torch.from_numpy(pcd.points).cuda()[None], num_pts).cpu().numpy()
            if pcd.time is not None:
                times = pcd.time[mask]
            else:
                times = None
            xyz = pcd.points[mask]
            rgb = pcd.colors[mask]
            normals = pcd.normals[mask]
            if times is not None:
                time_mask = (times[:,0] < time_duration[1]) & (times[:,0] > time_duration[0])
                xyz = xyz[time_mask]
                rgb = rgb[time_mask]
                normals = normals[time_mask]
                times = times[time_mask]
            pcd = BasicPointCloud(points=xyz, colors=rgb, normals=normals, time=times)
            
        if num_extra_pts > 0:
            times = pcd.time
            xyz = pcd.points
            rgb = pcd.colors
            normals = pcd.normals
            bound_min, bound_max = xyz.min(0), xyz.max(0)
            radius = 60.0 # (bound_max - bound_min).mean() + 10
            phi = 2.0 * np.pi * np.random.rand(num_extra_pts)
            theta = np.arccos(2.0 * np.random.rand(num_extra_pts) - 1.0)
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            xyz_extra = np.stack([x, y, z], axis=1)
            normals_extra = np.zeros_like(xyz_extra)
            rgb_extra = np.ones((num_extra_pts, 3)) / 2
            
            xyz = np.concatenate([xyz, xyz_extra], axis=0)
            rgb = np.concatenate([rgb, rgb_extra], axis=0)
            normals = np.concatenate([normals, normals_extra], axis=0)
            
            if times is not None:
                times_extra = torch.zeros(((num_extra_pts, 3))) + (time_duration[0] + time_duration[1]) / 2
                times = np.concatenate([times, times_extra], axis=0)
                
            pcd = BasicPointCloud(points=xyz, 
                                colors=rgb,
                                normals=normals,
                                time=times)
    else:
        pcd = None
        ply_path_origin = None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path_origin)
    return scene_info

class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = open(filename, 'w')
            self.fs.write('%YAML:1.0\r\n')
            self.fs.write('---\r\n')
        else:
            assert os.path.exists(filename), filename
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.isWrite = isWrite

    def __del__(self):
        if self.isWrite:
            self.fs.close()
        else:
            cv2.FileStorage.release(self.fs)

    def _write(self, out):
        self.fs.write(out + '\r\n')

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            self._write('{}: !!opencv-matrix'.format(key))
            self._write('  rows: {}'.format(value.shape[0]))
            self._write('  cols: {}'.format(value.shape[1]))
            self._write('  dt: d')
            self._write('  data: [{}]'.format(', '.join(['{:.10f}'.format(i) for i in value.reshape(-1)])))
        elif dt == 'list':
            self._write('{}:'.format(key))
            for elem in value:
                self._write('  - "{}"'.format(elem))
        elif dt == 'real':
            if isinstance(value, np.ndarray):
                value = value.item()
            self._write('{}: {:.10f}'.format(key, value))  # as accurate as possible
        else:
            raise NotImplementedError

    def read(self, key, dt='mat'):
        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        elif dt == 'real':
            output = self.fs.getNode(key).real()
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)

def readZJUCameras(path, white_background, cam_names=[], time_duration=None, frame_ratio=1, dataloader=False):
    extri_path = os.path.join(path, 'extri.yml')
    intri_path = os.path.join(path, 'intri.yml')
    assert os.path.exists(intri_path), intri_path
    assert os.path.exists(extri_path), extri_path

    intri = FileStorage(intri_path)
    extri = FileStorage(extri_path)
    cam_infos = []
    cam_names = intri.read('names', dt='list')
    idx = 0
    cams = []
    frames = int(extri.read('frames', dt='real'))
    with open(os.path.join(path, 'sync.json'), 'r') as f:
        sync_json = json.load(f)
    
    for cam in cam_names:
        if cam in ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0014', '0018', '0019', '0020', '0021', '0022', '0023']:
            continue
        # Intrinsics
        K = intri.read('K_{}'.format(cam))
        H = int(intri.read('H_{}'.format(cam), dt='real')) or -1
        W = int(intri.read('W_{}'.format(cam), dt='real')) or -1
        invK = np.linalg.inv(K)

        # Extrinsics
        Tvec = extri.read('T_{}'.format(cam))
        Rvec = extri.read('R_{}'.format(cam))
        if Rvec is not None: R = cv2.Rodrigues(Rvec)[0]
        else:
            R = extri.read('Rot_{}'.format(cam))
            Rvec = cv2.Rodrigues(R)[0]
        RT = np.hstack((R, Tvec))

        R = R
        T = Tvec
        # C = - Rvec.T @ Tvec
        # RT = RT
        # Rvec = Rvec
        # P = K @ RT

        w2c = np.vstack((RT, np.array([0, 0, 0, 1])))
        # c2w = np.linalg.inv(w2c)
        
        # cams.append((c2w[:3, 0] + c2w[:3, 3], c2w[:3, 1] + c2w[:3, 3], c2w[:3, 2] + c2w[:3, 3], c2w[:3, 3]))
        # get the world-to-camera transform and set R, T
        # w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        
        # Distortion
        D = intri.read('D_{}'.format(cam))
        if D is None: D = intri.read('dist_{}'.format(cam))
        D = D
        distort = D
        # distort = np.array(cam_data[cam_id]['distCoeff'][0:2]+cam_data[cam_id]['distCoeff'][3:5])
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, distort, (W, H), 1, (W, H))
        mapx, mapy = cv2.initUndistortRectifyMap(K, distort, None, new_camera_matrix, (W, H), cv2.CV_32FC1)
        K = new_camera_matrix
        # intr_mat[:2] /= 2
        # intr_mats.append(new_camera_matrix)
        
        # distortions.append((mapx, mapy))
        # # Time input
        # t = extri.read('t_{}'.format(cam), dt='real') or 0  # temporal index, might all be 0
        # v = extri.read('v_{}'.format(cam), dt='real') or 0  # temporal index, might all be 0

        # # Bounds, could be overwritten
        # n = extri.read('n_{}'.format(cam), dt='real') or 0.0001  # temporal index, might all be 0
        # f = extri.read('f_{}'.format(cam), dt='real') or 1e6  # temporal index, might all be 0
        # bounds = extri.read('bounds_{}'.format(cam))
        # bounds = np.array([[-1e6, -1e6, -1e6], [1e6, 1e6, 1e6]]) if bounds is None else bounds

        # # CCM
        # ccm = intri.read('ccm_{}'.format(cam))
        # ccm = np.eye(3) if ccm is None else ccm
        
        FovX = np.arctan2(W / 2.0, K[0, 0]) * 2.0
        FovY = np.arctan2(H / 2.0, K[1, 1]) * 2.0
        # read json file from a.json
            
        for ii in range(0, frames, 1):
            timestamp = ii / 60. - sync_json[cam] # Assuming 60 fps, adjust if needed
            if frame_ratio > 1:
                timestamp /= frame_ratio
            if time_duration is not None:
                if timestamp < time_duration[0]:
                    continue
                if timestamp > time_duration[1]:
                    break
            image_path = os.path.join(path, 'images', f'cam{cam}_{ii:06d}.jpg') # .replace('hdImgs_unditorted', 'hdImgs_unditorted_rgba').replace('.jpg', '.png')
            image_name = f'cam{cam}_{ii:06d}'
            
            if not dataloader:
                with Image.open(image_path) as image_load:
                    im_data = np.array(image_load.convert("RGBA"))

                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                if norm_data[:, :, 3:4].min() < 1:
                    arr = np.concatenate([arr, norm_data[:, :, 3:4]], axis=2)
                    image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")
                else:
                    image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            else:
                image = np.empty(0)
            
            cam_infos.append(CameraInfo(uid=idx + ii * len(cam_names), R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=None,
                            image_path=image_path, image_name=image_name, width=W, height=H, timestamp=timestamp,
                            fl_x=K[0, 0], fl_y=K[1, 1], cx=K[0, 2], cy=K[1, 2]))
        idx += 1
    # #save .obj
    # # if not os.path.exists(os.path.join(path, 'cams.obj')):
    # ii = 0
    # points = []
    # with open(os.path.join(path, 'cams.obj'), 'w') as f:
    #     for idx, cam in enumerate(cams):
    #         f.write(f"v {cam[3][0]} {cam[3][1]} {cam[3][2]} 1 1 1\n")
    #         f.write(f"v {cam[0][0]} {cam[0][1]} {cam[0][2]} 1 0 1\n")
    #         f.write(f"v {cam[1][0]} {cam[1][1]} {cam[1][2]} 0 1 0\n")
    #         f.write(f"v {cam[2][0]} {cam[2][1]} {cam[2][2]} 0 0 1\n")
            
    #     for ii in range(0, len(cams), 1):
    #         f.write(f"l {ii*4+1} {ii*4+2}\n")
    #         f.write(f"l {ii*4+1} {ii*4+3}\n")
    #         f.write(f"l {ii*4+1} {ii*4+4}\n")
    # exit()
    # # Average
    # avg_c2w_R = extri.read('avg_c2w_R')
    # avg_c2w_T = extri.read('avg_c2w_T')
    # if avg_c2w_R is not None: cams.avg_c2w_R = avg_c2w_R
    # if avg_c2w_T is not None: cams.avg_c2w_T = avg_c2w_T

    return cam_infos

def readZJUInfo(path, white_background, num_pts=100_000, time_duration=None, num_extra_pts=0, frame_ratio=1, dataloader=False):
    print("Reading Training Set")
    train_cam_infos = readZJUCameras(path, white_background, time_duration=time_duration, frame_ratio=frame_ratio, dataloader=dataloader)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    if pcd.points.shape[0] > num_pts:
        mask = np.random.randint(0, pcd.points.shape[0], num_pts)
        # mask = fps(torch.from_numpy(pcd.points).cuda()[None], num_pts).cpu().numpy()
        if pcd.time is not None:
            times = pcd.time[mask]
        else:
            times = None
        xyz = pcd.points[mask]
        rgb = pcd.colors[mask]
        normals = pcd.normals[mask]
        if times is not None:
            time_mask = (times[:,0] < time_duration[1]) & (times[:,0] > time_duration[0])
            xyz = xyz[time_mask]
            rgb = rgb[time_mask]
            normals = normals[time_mask]
            times = times[time_mask]
        pcd = BasicPointCloud(points=xyz, colors=rgb, normals=normals, time=times)
        
    if num_extra_pts > 0:
        times = pcd.time
        xyz = pcd.points
        rgb = pcd.colors
        normals = pcd.normals
        bound_min, bound_max = xyz.min(0), xyz.max(0)
        radius = 60.0 # (bound_max - bound_min).mean() + 10
        phi = 2.0 * np.pi * np.random.rand(num_extra_pts)
        theta = np.arccos(2.0 * np.random.rand(num_extra_pts) - 1.0)
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        xyz_extra = np.stack([x, y, z], axis=1)
        normals_extra = np.zeros_like(xyz_extra)
        rgb_extra = np.ones((num_extra_pts, 3)) / 2
        
        xyz = np.concatenate([xyz, xyz_extra], axis=0)
        rgb = np.concatenate([rgb, rgb_extra], axis=0)
        normals = np.concatenate([normals, normals_extra], axis=0)
        
        if times is not None:
            times_extra = torch.zeros(((num_extra_pts, 3))) + (time_duration[0] + time_duration[1]) / 2
            times = np.concatenate([times, times_extra], axis=0)
            
        pcd = BasicPointCloud(points=xyz, 
                              colors=rgb,
                              normals=normals,
                              time=times)
        
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=train_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readTHUCameras(path, white_background, cam_names=[], time_duration=None, frame_ratio=1, dataloader=False):
    with open(os.path.join(path, 'calibration_full.json'), 'r') as f:
        calib = json.load(f)
        
    poses = []
    Ks = []
    mapxy = []
    cam_names = []
    for k in calib.keys():
        # if int(k) > 53:
        #     continue
        cam_names.append(k)
        RT = np.eye(4)
        RT[:3, :3] = np.array(calib[k]['R']).reshape((3,3))
        RT[:3, 3] = np.array(calib[k]['T'])
        # RT = np.linalg.inv(RT)  # convert to world to camera
        W, H = calib[k]['imgSize'][0], calib[k]['imgSize'][1]
        poses.append(RT)
        K = np.array(calib[k]['K']).reshape((3,3))
        D = np.array(calib[k]['distCoeff'])
        
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, D, (W, H), 1, (W, H))
        mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_camera_matrix, (W, H), cv2.CV_32FC1)
        
        new_camera_matrix[0, 2] -= 100
        new_camera_matrix[1, 2] -= 100
        
        W, H = W - 200, H - 200
        new_camera_matrix[:2] /= 2
        W, H = W // 2, H // 2
        Ks.append(new_camera_matrix)
        mapxy.append((mapx, mapy))
    idx = 0
    cam_infos = []
    frames = 10000
    # with open(os.path.join(path, 'sync.json'), 'r') as f:
    #     sync_json = json.load(f)
    
    for index_ii, cam in enumerate(cam_names):
        # if cam in ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0014', '0018', '0019', '0020', '0021', '0022', '0023']:
        #     continue
        # Intrinsics
        aa = list(range(1, 61, 1))
        aa += list(range(3, 61, 9))
        if int(cam) not in aa or int(cam) == 28:
            continue
        if int(cam) in [14, 16, 19, 20, 21, 22, 24, 25, 26, 28, 29, 25, 35, 42, 43, 45]:
            continue
        # if int(cam) == 22:
        #     continue
        # if int(cam) == 42:
        #     continue
        K = Ks[index_ii]

        # Extrinsics
        RT = poses[index_ii][:3]

        R = RT[:3 ,:3]
        T = RT[:3, 3]
        # C = - Rvec.T @ Tvec
        # RT = RT
        # Rvec = Rvec
        # P = K @ RT

        w2c = np.vstack((RT, np.array([0, 0, 0, 1])))
        # c2w = np.linalg.inv(w2c)
        
        # cams.append((c2w[:3, 0] + c2w[:3, 3], c2w[:3, 1] + c2w[:3, 3], c2w[:3, 2] + c2w[:3, 3], c2w[:3, 3]))
        # get the world-to-camera transform and set R, T
        # w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        
        # Distortion
        mapx, mapy = mapxy[index_ii][0], mapxy[index_ii][1]
        
        FovX = np.arctan2(W / 2.0, K[0, 0]) * 2.0
        FovY = np.arctan2(H / 2.0, K[1, 1]) * 2.0
        # read json file from a.json
        fps = 50. # Assuming 50 fps, adjust if needed
        for ii in range(0, frames, 1):
            # if ii < 1000 or ii > 1150:
            #     continue
            timestamp = ii / fps
            if frame_ratio > 1:
                timestamp /= frame_ratio
            if time_duration is not None:
                if timestamp < time_duration[0]:
                    continue
                if timestamp > time_duration[1]:
                    break
            # image_path = os.path.join(path, 'images', f'cam{cam}_{ii:06d}.jpg') # .replace('hdImgs_unditorted', 'hdImgs_unditorted_rgba').replace('.jpg', '.png')
            image_path = os.path.join('/media/bbnc/Elements/abuzabi', cam, f'cam{cam}_{ii:06d}.jpg')
            image_name = f'cam{cam}_{ii:06d}'
            
            if not dataloader:
                with Image.open(image_path) as image_load:
                    im_data = np.array(image_load.convert("RGBA"))

                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                if norm_data[:, :, 3:4].min() < 1:
                    arr = np.concatenate([arr, norm_data[:, :, 3:4]], axis=2)
                    image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")
                else:
                    image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            else:
                image = np.empty(0)
            
            cam_infos.append(CameraInfo(uid=idx + ii * len(cam_names), R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=None,
                            image_path=image_path, image_name=image_name, width=W, height=H, timestamp=(timestamp - time_duration[0])/1,
                            fl_x=K[0, 0], fl_y=K[1, 1], cx=K[0, 2], cy=K[1, 2]))
        idx += 1

    return cam_infos

def readTHUInfo(path, white_background, num_pts=100_000, time_duration=None, num_extra_pts=0, frame_ratio=1, dataloader=False):
    print("Reading Training Set")
    train_cam_infos = readTHUCameras(path, white_background, time_duration=time_duration, frame_ratio=frame_ratio, dataloader=dataloader)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    if pcd.points.shape[0] > num_pts:
        mask = np.random.randint(0, pcd.points.shape[0], num_pts)
        # mask = fps(torch.from_numpy(pcd.points).cuda()[None], num_pts).cpu().numpy()
        if pcd.time is not None:
            times = pcd.time[mask]
        else:
            times = None
        xyz = pcd.points[mask]
        rgb = pcd.colors[mask]
        normals = pcd.normals[mask]
        if times is not None:
            time_mask = (times[:,0] < time_duration[1]) & (times[:,0] > time_duration[0])
            xyz = xyz[time_mask]
            rgb = rgb[time_mask]
            normals = normals[time_mask]
            times = times[time_mask]
        pcd = BasicPointCloud(points=xyz, colors=rgb, normals=normals, time=times)
        
    if num_extra_pts > 0:
        times = pcd.time
        xyz = pcd.points
        rgb = pcd.colors
        normals = pcd.normals
        bound_min, bound_max = xyz.min(0), xyz.max(0)
        radius = 60.0 # (bound_max - bound_min).mean() + 10
        phi = 2.0 * np.pi * np.random.rand(num_extra_pts)
        theta = np.arccos(2.0 * np.random.rand(num_extra_pts) - 1.0)
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        xyz_extra = np.stack([x, y, z], axis=1)
        normals_extra = np.zeros_like(xyz_extra)
        rgb_extra = np.ones((num_extra_pts, 3)) / 2
        
        xyz = np.concatenate([xyz, xyz_extra], axis=0)
        rgb = np.concatenate([rgb, rgb_extra], axis=0)
        normals = np.concatenate([normals, normals_extra], axis=0)
        
        if times is not None:
            times_extra = torch.zeros(((num_extra_pts, 3))) + (time_duration[0] + time_duration[1]) / 2
            times = np.concatenate([times, times_extra], axis=0)
            
        pcd = BasicPointCloud(points=xyz, 
                              colors=rgb,
                              normals=normals,
                              time=times)
        
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=train_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readNerfiesInfo(path, eval):
    print("Reading Nerfies Info")
    cam_infos, train_num, scene_center, scene_scale = readNerfiesCameras(path)

    if eval:
        train_cam_infos = cam_infos[:train_num]
        test_cam_infos = cam_infos[train_num:]
        # train_cam_infos = cam_infos[:train_num]
        # test_cam_infos = cam_infos[:train_num]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        print(f"Generating point cloud from nerfies...")

        xyz = np.load(os.path.join(path, "points.npy"))
        xyz = (xyz - scene_center) * scene_scale
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readNerfiesCameras(path):
    with open(f'{path}/scene.json', 'r') as f:
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)

    coord_scale = scene_json['scale']
    scene_center = scene_json['center']

    name = path.split('/')[-2]
    if name.startswith('vrig'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 0.5
    elif name.startswith('NDS'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 1.0
    elif name.startswith('interp'):
        all_id = dataset_json['ids']
        train_img = all_id[::4]
        val_img = all_id[2::4]
        all_img = train_img + val_img
        ratio = 0.5
    else:  # for hypernerf
        train_img = dataset_json['ids'][::4]
        all_img = train_img
        ratio = 0.5

    train_num = len(train_img)

    all_cam = [meta_json[i]['camera_id'] for i in all_img]
    all_time = [meta_json[i]['time_id'] for i in all_img]
    max_time = max(all_time)
    all_time = [meta_json[i]['time_id'] / 30 for i in all_img]
    selected_time = set(all_time)

    # all poses
    all_cam_params = []
    for im in all_img:
        camera = camera_nerfies_from_JSON(f'{path}/camera/{im}.json', ratio)
        camera['position'] = camera['position'] - scene_center
        camera['position'] = camera['position'] * coord_scale
        all_cam_params.append(camera)

    all_img = [f'{path}/rgb/{int(1 / ratio)}x/{i}.png' for i in all_img]

    cam_infos = []
    for idx in range(len(all_img)):
        image_path = all_img[idx]
        image = np.array(Image.open(image_path))
        image = Image.fromarray((image).astype(np.uint8))
        image_name = Path(image_path).stem

        orientation = all_cam_params[idx]['orientation'].T
        position = -all_cam_params[idx]['position'] @ orientation
        focal = all_cam_params[idx]['focal_length']
        fid = all_time[idx]
        T = position
        R = orientation

        FovY = focal2fov(focal, image.size[1])
        FovX = focal2fov(focal, image.size[0])
        depth = None
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth,
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              timestamp=fid)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos, train_num, scene_center, coord_scale

def camera_nerfies_from_JSON(path, scale):
    """Loads a JSON camera into memory."""
    with open(path, 'r') as fp:
        camera_json = json.load(fp)

    # Fix old camera JSON.
    if 'tangential' in camera_json:
        camera_json['tangential_distortion'] = camera_json['tangential']

    return dict(
        orientation=np.array(camera_json['orientation']),
        position=np.array(camera_json['position']),
        focal_length=camera_json['focal_length'] * scale,
        principal_point=np.array(camera_json['principal_point']) * scale,
        skew=camera_json['skew'],
        pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
        radial_distortion=np.array(camera_json['radial_distortion']),
        tangential_distortion=np.array(camera_json['tangential_distortion']),
        image_size=np.array((int(round(camera_json['image_size'][0] * scale)),
                             int(round(camera_json['image_size'][1] * scale)))),
    )


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    'THU': readTHUInfo,
    'Nerfies': readNerfiesInfo
}