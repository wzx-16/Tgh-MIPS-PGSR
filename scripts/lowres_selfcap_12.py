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

    #image_folders_path = os.path.join(args.path, "images_folder/")
    images_path = os.path.join(args.path, "images_full_res/")
    #os.makedirs(images_path, exist_ok=True)
    
    # for video in videos:
    #     cam_name = video.split('/')[-1].split('.')[-2]
    #     do_system(f"ffmpeg -i {video} -start_number 0 {images_path}/{cam_name}_%04d.png")
        
    # load data
    #base_images_path = os.path.join(args.path, "mattings/")

    # Load images from camera-specific subfolders
    # images = []
    # cams = []
    # for cam_folder in sorted(glob.glob(os.path.join(image_folders_path, "*"))):
    #     if os.path.isdir(cam_folder):
    #         cam_id = os.path.basename(cam_folder)
    #         cam_images = [f[len(args.path):] for f in sorted(glob.glob(os.path.join(cam_folder, "*"))) 
    #                      if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    #         images.extend(cam_images)
    #         cams.append(cam_id)
    # cams = sorted(set(cams))
    # print(f"Images: {images}")
    # print(f"Cameras: {cams}")
    images = [f[len(args.path):] for f in sorted(glob.glob(os.path.join(args.path, "images_undist/", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
    print(f"Found {len(images)} images: {images}")
    # images = [im for im in images if int(im[11:17]) < 200]
    # cams = sorted(set([im[7:10] for im in images]))
    # print(images)
    # print(cams)
    # exit()
    


    
    # with open(os.path.join(args.path, 'calibration_full.json'), 'r') as f:
    #     calib = json.load(f)
    # #poses_bounds = np.load(os.path.join(args.path, 'poses_bounds.npy'))
    # #N = poses_bounds.shape[0]
    # cameras = calib
    # #camera_poses = calib["camera_poses"]
    # N = len(cameras.keys())
    # poses = []
    # Ks = []
    # #mapxy = []
    # cam_names = []
    # for k in cameras.keys():
    #     # if int(k) > 53:
    #     #     continue
    #     cam_names.append(k)
    #     RT = np.eye(4)
    #     RT[:3, :3] = np.array(cameras[k]['R']).reshape((3,3))
    #     RT[:3, 3] = np.array(cameras[k]['T'])
    #     RT = np.linalg.inv(RT)  # convert to world to camera
    #     W, H = cameras[k]['imgSize'][0], cameras[k]['imgSize'][1]
    #     W = W
    #     H = H
    #     poses.append(RT)
    #     K = np.array(cameras[k]['K']).reshape((3,3))
    #     # K[0][0] /= 2
    #     # K[0][2] /= 2
    #     # K[1][1] /= 2
    #     # K[1][2] /= 2
    #     D = np.array(cameras[k]['distCoeff'])
        
        #new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, D, (W, H), 1, (W, H))
        #mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_camera_matrix, (W, H), cv2.CV_32FC1)
    for imagename in images:
        #imname = imagename[14:17] + "_" + imagename[18:]
        undis_img = cv2.imread(os.path.join(args.path, imagename), cv2.IMREAD_UNCHANGED)
        #undis_img = cv2.remap(undis_img, mapx, mapy, cv2.INTER_LINEAR)
        #print(undis_img.shape)
        #undis_img = cv2.resize(undis_img, (undis_img.shape[1] // 2, undis_img.shape[0] // 2), interpolation=cv2.INTER_AREA)
        #undis_img = undis_img[90:2070, 90:3750]
        undis_img = cv2.resize(undis_img, (undis_img.shape[1] // 12, undis_img.shape[0] // 12), interpolation=cv2.INTER_AREA)
        #print(os.path.join(images_path, imname))
        #print(os.path.join(os.path.join(args.path, "images"), imname))
        #print(os.path.join(os.path.join(args.path, "images_450"), imname))
        #cv2.imwrite(os.path.join(os.path.join(args.path, "images_450"), imname), undis_img)
        #print(args.path)
        #print(os.path.join(os.path.join(args.path, "images"), imagename[19:]))
        print(os.path.join(os.path.join(args.path, "images"), imagename[14:]))
        cv2.imwrite(os.path.join(os.path.join(args.path, "images"), imagename[14:]), undis_img)

        # for imagename in images:
        #     if imagename[18:21] == k:
        #         undis_img = cv2.imread(os.path.join(args.path, imagename), cv2.IMREAD_UNCHANGED)
        #         #undis_img = cv2.remap(undis_img, mapx, mapy, cv2.INTER_LINEAR)
        #         undis_img = cv2.resize(undis_img, (undis_img.shape[1] // 2, undis_img.shape[0] // 2), interpolation=cv2.INTER_AREA)
        #         #print(undis_img.shape)
        #         #undis_img = undis_img[200:3800, 100:2900]
        #         #print(os.path.join(args.path, imagename))
        #         #print(os.path.join(os.path.join(args.path, "mattings"), imagename[18:]))
        #         os.makedirs(os.path.join(os.path.join(args.path, "mattings"), imagename[18:21]), exist_ok=True)
        #         cv2.imwrite(os.path.join(os.path.join(args.path, "mattings"), imagename[18:]), undis_img)
        # new_camera_matrix[0, 2] -= 120
        # new_camera_matrix[1, 2] -= 180
        
        # W, H = W - 240, H - 360

        # new_camera_matrix[0, 0] /= 2
        # new_camera_matrix[1, 1] /= 2
        # new_camera_matrix[0, 2] /= 2
        # new_camera_matrix[1, 2] /= 2
        
        # Ks.append(K)
        # print(k)
        #mapxy.append((mapx, mapy))