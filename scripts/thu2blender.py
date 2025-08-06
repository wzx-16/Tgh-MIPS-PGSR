import os
import argparse
import glob
import cv2

import numpy as np
import json
from PIL import Image


def closest_point_2_lines(oa, da, ob, db): 
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

if __name__ == '__main__':
    parser = argparse.ArgumentParser() # TODO: refine it.
    parser.add_argument("path", default="", help="input path to the video")
    #parser.add_argument("frame_num", type=str, default="")
    args = parser.parse_args()

    # path must end with / to make sure image path is relative
    if args.path[-1] != '/':
        args.path += '/'
    #frame_num = args.frame_num
        
    # extract images
    #videos = [os.path.join(args.path, vname) for vname in os.listdir(args.path) if vname.endswith(".mp4")]
    images_path = os.path.join(args.path, "images/")
    #os.makedirs(images_path, exist_ok=True)
    
    # for video in videos:
    #     cam_name = video.split('/')[-1].split('.')[-2]
    #     do_system(f"ffmpeg -i {video} -start_number 0 {images_path}/{cam_name}_%04d.png")
        
    # load data
    images = [f[len(args.path):] for f in sorted(glob.glob(os.path.join(args.path, "images/", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
    images = [im for im in images if int(im[11:17]) < 200]
    cams = sorted(set([im[7:10] for im in images]))
    #print(images)
    print(cams)
    


    
    with open(os.path.join(args.path, 'calibration_full.json'), 'r') as f:
        calib = json.load(f)
    #poses_bounds = np.load(os.path.join(args.path, 'poses_bounds.npy'))
    #N = poses_bounds.shape[0]
    cameras = calib["cameras"]
    camera_poses = calib["camera_poses"]
    N = len(cameras.keys())
    poses = []
    Ks = []
    #mapxy = []
    cam_names = []
    for k in cameras.keys():
        # if int(k) > 53:
        #     continue
        cam_names.append(k)
        RT = np.eye(4)
        RT[:3, :3] = np.array(camera_poses[k]['R'])
        RT[:3, 3] = np.array(camera_poses[k]['T']).reshape(3)
        RT = np.linalg.inv(RT)  # convert to world to camera
        W, H = cameras[k]['image_size'][0], cameras[k]['image_size'][1]
        poses.append(RT)
        K = np.array(cameras[k]['K'])
        #D = np.array(cameras[k]['distCoeff'])
        
        #new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, D, (W, H), 1, (W, H))
        # mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_camera_matrix, (W, H), cv2.CV_32FC1)
        
        # new_camera_matrix[0, 2] -= 120
        # new_camera_matrix[1, 2] -= 180
        
        # W, H = W - 240, H - 360

        # new_camera_matrix[0, 0] /= 2
        # new_camera_matrix[1, 1] /= 2
        # new_camera_matrix[0, 2] /= 2
        # new_camera_matrix[1, 2] /= 2
        
        Ks.append(K)
        print(k)
        #mapxy.append((mapx, mapy))


    # for imagename in images:
    #     filename = imagename.split("/")[1]
    #     if filename.endswith(".jpg"):
    #         full_path = os.path.join(images_path, filename)
    #         img = Image.open(full_path)
    #         # left = 120
    #         # top = 180
    #         # right = left + W
    #         # bottom = top + H
    #         # cropped_img = img.crop((left, top, right, bottom))
    #         # resized_img = cropped_img.resize((cropped_img.width // 2, cropped_img.height // 2))
    #         resized_img = img.resize((img.width // 2, img.height // 2))
    #         resized_img.save(full_path)
    #         resized_img.save(full_path)
    #         print(filename)


    frames = 10000
    # with open(os.path.join(path, 'sync.json'), 'r') as f:
    #     sync_json = json.load(f)
    
    # for index_ii, cam in enumerate(cam_names):
    #     # if cam in ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0014', '0018', '0019', '0020', '0021', '0022', '0023']:
    #     #     continue
    #     # Intrinsics
    #     aa = list(range(1, 61, 1))
    #     # aa += list(range(3, 61, 9))
    #     if int(cam) not in aa or int(cam) == 28 or int(cam) == 30:
    #         continue
    #     if int(cam) == 22:
    #         continue
    #     K = Ks[index_ii]

    #     # Extrinsics
    #     RT = poses[index_ii][:3]

    #     R = RT[:3 ,:3]
    #     T = RT[:3, 3]
    #     # C = - Rvec.T @ Tvec
    #     # RT = RT
    #     # Rvec = Rvec
    #     # P = K @ RT

    #     w2c = np.vstack((RT, np.array([0, 0, 0, 1])))
    #     # c2w = np.linalg.inv(w2c)
        
    #     # cams.append((c2w[:3, 0] + c2w[:3, 3], c2w[:3, 1] + c2w[:3, 3], c2w[:3, 2] + c2w[:3, 3], c2w[:3, 3]))
    #     # get the world-to-camera transform and set R, T
    #     # w2c = np.linalg.inv(c2w)
    #     #R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    #     T = w2c[:3, 3]
        
    #     # Distortion
    #     #mapx, mapy = mapxy[index_ii][0], mapxy[index_ii][1]
        
    #     #FovX = np.arctan2(W / 2.0, K[0, 0]) * 2.0
    #     #FovY = np.arctan2(H / 2.0, K[1, 1]) * 2.0
    #     # read json file from a.json
    #     frame_ratio = 1
    #     dataloader = False
    #     time_duration = [0, 10]
    #     white_background = False
    #     for ii in range(0, frames, 1):
    #         timestamp = ii / 30. # Assuming 30 fps, adjust if needed
    #         if frame_ratio > 1:
    #             timestamp /= frame_ratio
    #         if time_duration is not None:
    #             if timestamp < time_duration[0]:
    #                 continue
    #             if timestamp > time_duration[1]:
    #                 break
    #         # image_path = os.path.join(path, 'images', f'cam{cam}_{ii:06d}.jpg') # .replace('hdImgs_unditorted', 'hdImgs_unditorted_rgba').replace('.jpg', '.png')
    #         image_path = os.path.join('/data1/zhanfengliao/projects/4D-Rotor-Gaussians/data/custom/abuzabi/images', cam, f'cam{cam}_{ii:06d}.jpg')
    #         image_name = f'cam{cam}_{ii:06d}'
            
    #         if not dataloader:
    #             with Image.open(image_path) as image_load:
    #                 im_data = np.array(image_load.convert("RGBA"))

    #             bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

    #             norm_data = im_data / 255.0
    #             arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
    #             if norm_data[:, :, 3:4].min() < 1:
    #                 arr = np.concatenate([arr, norm_data[:, :, 3:4]], axis=2)
    #                 image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")
    #             else:
    #                 image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
    #         else:
    #             image = np.empty(0)
            

    print(f'[INFO] loaded {len(images)} images from {len(cams)} videos, {N} poses_bounds as ')

    assert N == len(cams)

    # poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N, 3, 5)
    # bounds = poses_bounds[:, -2:] # (N, 2)

    #H, W, fl = poses[0, :, -1] 

    #print(f'[INFO] H = {H}, W = {W}, fl = {fl}')
    poses = np.stack(poses, axis = 0)
    # inversion of this: https://github.com/Fyusion/LLFF/blob/c6e27b1ee59cb18f054ccb0f87a90214dbe70482/llff/poses/pose_utils.py#L51
    #poses = np.concatenate([poses[..., 1:2], poses[..., 0:1], -poses[..., 2:3], poses[..., 3:4]], -1) # (N, 3, 4)
    # to homogeneous 
    # last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N, 1, 4)
    # poses = np.concatenate([poses, last_row], axis=1) # (N, 4, 4) 

    # the following stuff are from colmap2nerf... 
    #poses = np.stack(poses, axis = 0)
    poses[:, 0:3, 1] *= -1
    poses[:, 0:3, 2] *= -1
    #poses = poses[:, [1, 0, 2, 3], :] # swap y and z
    #poses[:, 2, :] *= -1 # flip whole world upside down

    # up = poses[:, 0:3, 1].sum(0)
    # up = up / np.linalg.norm(up)
    # R = rotmat(up, [0, 0, 1]) # rotate up vector to [0,0,1]
    # R = np.pad(R, [0, 1])
    # R[-1, -1] = 1

    # poses = R @ poses

    # totw = 0.0
    # totp = np.array([0.0, 0.0, 0.0])
    # for i in range(N):
    #     mf = poses[i, :3, :]
    #     for j in range(i + 1, N):
    #         mg = poses[j, :3, :]
    #         p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
    #         #print(i, j, p, w)
    #         if w > 0.01:
    #             totp += p * w
    #             totw += w
    # totp /= totw
    # print(f'[INFO] totp = {totp}')
    # poses[:, :3, 3] -= totp

    # avglen = np.linalg.norm(poses[:, :3, 3], axis=-1).mean()

    # poses[:, :3, 3] *= 4.0 / avglen

    #print(f'[INFO] average radius = {avglen}')
    
    train_frames = []
    test_frames = []
    for i in range(N):
        cam_frames = [{'file_path': im.lstrip("/").split('.')[0], 
                       'fl_x': Ks[i][0, 0] / 5,
                       'fl_y': Ks[i][1, 1] / 5,
                       'cx': Ks[i][0, 2] / 5,
                       'cy': Ks[i][1, 2] / 5,
                       'transform_matrix': poses[i].tolist(),
                       'time': int(im.lstrip("/").split('.')[0][-4:]) / 30.} for im in images if cams[i] == im[7:10]]
        if i == 0:
            test_frames += cam_frames
        else:
            train_frames += cam_frames

    train_transforms = {
        'w': W // 5,
        'h': H // 5,
        'fl_x': Ks[0][0,0] / 5,
        'fl_y': Ks[0][1,1] / 5,
        'cx': Ks[0][0,2] / 5,
        'cy': Ks[0][1,2] / 5,
        'frames': train_frames,
    }
    test_transforms = {
        'w': W // 5,
        'h': H // 5,
        'fl_x': Ks[0][0,0] / 5,
        'fl_y': Ks[0][1,1] / 5,
        'cx': Ks[0][0,2] / 5,
        'cy': Ks[0][1,2] / 5,
        'frames': test_frames,
    }

    train_output_path = os.path.join(args.path, 'transforms_train.json')
    test_output_path = os.path.join(args.path, 'transforms_test.json')
    print(f'[INFO] write to {train_output_path} and {test_output_path}')
    with open(train_output_path, 'w') as f:
        json.dump(train_transforms, f, indent=2)
    with open(test_output_path, 'w') as f:
        json.dump(test_transforms, f, indent=2)