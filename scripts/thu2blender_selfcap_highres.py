#!/usr/bin/env python3
"""
Create NeRF-style transforms_train.json and transforms_test.json from images named
camid_frameid.jpg / .jpeg / .png and a calibration JSON like calibration_new_with_k.json.

This script supports calibration JSON in this form:
{
  "22139906": {
    "K": [[...], [...], [...]],
    "R": [[...], [...], [...]],
    "T": [...],
    "distCoeff": [[k1, k2, p1, p2, k3]],
    "imgSize": [2048, 1500],
    "rectifyAlpha": 1.0
  },
  ...
}

By default, image names like cam01_00000.jpg are mapped to calibration cameras by
JSON order: cam01 -> first calibration entry, cam02 -> second entry, etc.
If your file names use the actual serial number, e.g. 22139906_00000.jpg, pass:
  --filename_cam_mode serial
"""

import argparse
import glob
import json
import os
import re
from collections import OrderedDict

import cv2
import numpy as np


def parse_image_filename(filename, filename_cam_mode="index"):
    """
    Supported examples:
      index mode:  cam01_00001.jpg -> (1, 1)
                   cam1_00001.jpg  -> (1, 1)
      serial mode: 22139906_00001.jpg -> ("22139906", 1)
    """
    if filename_cam_mode == "serial":
        m = re.match(r"^(\d+)_(\d+)\.(jpg|jpeg|png)$", filename, re.IGNORECASE)
        if not m:
            return None, None
        return m.group(1), int(m.group(2))

    m = re.match(r"^cam(\d+)_(\d+)\.(jpg|jpeg|png)$", filename, re.IGNORECASE)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def load_calib_and_compute_demo_intrinsics(
    calib_path,
    alpha=None,
    crop_x=152,
    crop_y=150,
    crop_w=None,
    crop_h=None,
    filename_cam_mode="index",
):
    with open(calib_path, "r") as f:
        calib = json.load(f, object_pairs_hook=OrderedDict)

    if not isinstance(calib, dict):
        raise ValueError(f"Expected top-level JSON object in {calib_path}")

    # Support both the old list format and the uploaded serial-number-keyed format.
    if "calibrations" in calib:
        entries = []
        for item in calib["calibrations"]:
            cam_id = str(item["cameraSN"])
            K = np.array(
                [[item["fx"], 0.0, item["cx"]], [0.0, item["fy"], item["cy"]], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            D = np.array(
                [item.get("k1", 0.0), item.get("k2", 0.0), item.get("p1", 0.0), item.get("p2", 0.0), item.get("k3", 0.0)],
                dtype=np.float64,
            )
            W = int(item["imageSize"]["w"])
            H = int(item["imageSize"]["h"])

            if "world2Cam" not in item:
                raise ValueError("Old calibration-list format needs 'world2Cam' for pose generation.")
            w2c_info = item["world2Cam"]
            R_w2c = quaternion_to_rotation_matrix(
                float(w2c_info["qw"]), float(w2c_info["qx"]), float(w2c_info["qy"]), float(w2c_info["qz"])
            )
            t_w2c = np.array([w2c_info["x"], w2c_info["y"], w2c_info["z"]], dtype=np.float64)
            entries.append((cam_id, K, D, W, H, R_w2c, t_w2c, float(item.get("rectifyAlpha", 1.0))))
    else:
        entries = []
        for cam_id, item in calib.items():
            if not isinstance(item, dict) or "K" not in item:
                continue
            K = np.array(item["K"], dtype=np.float64)
            K = K.reshape((3, 3))
            D = np.array(item.get("distCoeff", [[0, 0, 0, 0, 0]]), dtype=np.float64).reshape(-1)
            W, H = map(int, item["imgSize"])
            R_c2w = np.array(item.get("R", np.eye(3)), dtype=np.float64)
            R_c2w = R_c2w.reshape((3, 3))
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = R_c2w
            c2w[:3, 3] = np.array(item.get("T", [0, 0, 0]), dtype=np.float64).reshape(3)
            w2c = np.linalg.inv(c2w)
            R_w2c = w2c[:3, :3]
            t_w2c = w2c[:3, 3]
            #t_w2c = np.array(item.get("T", [0, 0, 0]), dtype=np.float64).reshape(3)
            entries.append((str(cam_id), K, D, W, H, R_w2c, t_w2c, float(item.get("rectifyAlpha", 1.0))))

    if not entries:
        raise ValueError(f"No usable camera calibrations found in {calib_path}")

    cameras = {}
    for idx, (serial, K, D, W, H, R_w2c, t_w2c, item_alpha) in enumerate(entries, start=1):
        use_alpha = item_alpha if alpha is None else alpha

        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c
        c2w = np.linalg.inv(w2c)
        #K = K.reshape((3, 3))
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, D, (W, H), use_alpha, (W, H))

        # Match your demo crop convention by shifting principal point.
        new_camera_matrix[0, 2] -= crop_x
        new_camera_matrix[1, 2] -= crop_y

        # If not specified, keep your original demo dimensions.
        out_w = int(crop_w) if crop_w is not None else int(W - 2 * crop_x)
        out_h = int(crop_h) if crop_h is not None else int(H - 2 * crop_y)

        cam_record = {
            "serial": serial,
            "K_demo": new_camera_matrix,
            "w": out_w,
            "h": out_h,
            "c2w": c2w,
        }

        # index mode maps cam01/cam1 to first JSON camera, cam02 to second, etc.
        cameras[idx] = cam_record
        # serial mode maps serial_frameid.jpg directly to the serial key.
        cameras[serial] = cam_record

    return cameras


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / np.linalg.norm(q)
    qw, qx, qy, qz = q
    return np.array(
        [
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
        ],
        dtype=np.float64,
    )


def convert_pose_like_demo(c2w):
    pose = c2w.copy()
    pose[:3, 1] *= -1
    pose[:3, 2] *= -1
    return pose


def collect_images(images_dir):
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    return sorted(set(image_files))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".", help="Dataset root")
    parser.add_argument("--calib_json", default="extr.json", help="Calibration JSON path")
    parser.add_argument("--images_dir", default="images", help="Image folder under data_dir")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS for time field")
    parser.add_argument("--frame_id", type=int, default=-1, help="Only include this frame id; use -1 for all frames")
    parser.add_argument(
        "--filename_cam_mode",
        choices=["index", "serial"],
        default="serial",
        help="index: cam01_00000.jpg maps by calibration JSON order. serial: 22139906_00000.jpg maps by serial key.",
    )
    parser.add_argument("--test_cam", default=None, help="Camera to put in test set, e.g. 26 in index mode or 22139906 in serial mode")
    parser.add_argument("--alpha", type=float, default=None, help="Override rectifyAlpha/getOptimalNewCameraMatrix alpha")
    parser.add_argument("--crop_x", type=float, default=152)
    parser.add_argument("--crop_y", type=float, default=150)
    parser.add_argument("--crop_w", type=int, default=None)
    parser.add_argument("--crop_h", type=int, default=None)
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    calib_path = args.calib_json if os.path.isabs(args.calib_json) else os.path.join(data_dir, args.calib_json)
    images_dir = args.images_dir if os.path.isabs(args.images_dir) else os.path.join(data_dir, args.images_dir)

    camera_dict = load_calib_and_compute_demo_intrinsics(
        calib_path,
        alpha=args.alpha,
        crop_x=args.crop_x,
        crop_y=args.crop_y,
        crop_w=args.crop_w,
        crop_h=args.crop_h,
        filename_cam_mode=args.filename_cam_mode,
    )

    image_files = collect_images(images_dir)
    if not image_files:
        raise RuntimeError(f"No images found in {images_dir}")

    train_frames = []
    test_frames = []
    used_cameras = set()
    first_cam = None

    test_cam_key = None
    if args.test_cam is not None:
        test_cam_key = args.test_cam if args.filename_cam_mode == "serial" else int(args.test_cam)
    #print(image_files)
    for img_path in image_files:
        filename = os.path.basename(img_path)
        cam_key, frame_idx = parse_image_filename(filename, args.filename_cam_mode)
        print(f"Processing {filename}: cam_key={cam_key}, frame_idx={frame_idx}")
        if cam_key is None:
            print(f"[SKIP] unsupported filename format: {filename}")
            continue
        if args.frame_id >= 0 and frame_idx != args.frame_id:
            continue
        if frame_idx != 59:
            continue
        if cam_key not in camera_dict:
            print(f"[SKIP] no calibration for {filename} -> {cam_key}")
            continue

        cam = camera_dict[cam_key]
        if first_cam is None:
            first_cam = cam

        K_demo = cam["K_demo"]
        pose = convert_pose_like_demo(cam["c2w"])
        rel_path = os.path.relpath(img_path, data_dir)
        rel_path_no_ext = os.path.splitext(rel_path)[0].replace("\\", "/")

        frame = {
            "file_path": rel_path_no_ext,
            "fl_x": float(K_demo[0, 0]) / 3,
            "fl_y": float(K_demo[1, 1]) / 3,
            "cx": float(K_demo[0, 2]) / 3,
            "cy": float(K_demo[1, 2]) / 3,
            "transform_matrix": pose.tolist(),
            "time": frame_idx / args.fps,
            #"time": 0.03333333333333333,
            "camera_serial": cam["serial"],
        }

        if test_cam_key is not None and cam_key == test_cam_key:
            test_frames.append(frame)
        else:
            train_frames.append(frame)
        used_cameras.add(cam["serial"])
    print(train_frames)
    if not train_frames and not test_frames:
        raise RuntimeError("No valid frames matched between images and calibration")
    if first_cam is None:
        raise RuntimeError("No valid camera found")

    representative_frame = train_frames[0] if train_frames else test_frames[0]
    common = {
        "w": int(first_cam["w"]) // 3,
        "h": int(first_cam["h"]) // 3,
        "fl_x": representative_frame["fl_x"],
        "fl_y": representative_frame["fl_y"],
        "cx": representative_frame["cx"],
        "cy": representative_frame["cy"],
    }

    train_transforms = {**common, "frames": train_frames}
    test_transforms = {**common, "frames": test_frames}

    train_output_path = os.path.join(data_dir, "transforms_train.json")
    test_output_path = os.path.join(data_dir, "transforms_test.json")

    with open(train_output_path, "w") as f:
        json.dump(train_transforms, f, indent=2)
    with open(test_output_path, "w") as f:
        json.dump(test_transforms, f, indent=2)

    print(f"[INFO] loaded {len(used_cameras)} used cameras from {calib_path}")
    print(f"[INFO] train frames: {len(train_frames)}, test frames: {len(test_frames)}")
    print(f"[INFO] wrote {train_output_path}")
    print(f"[INFO] wrote {test_output_path}")


if __name__ == "__main__":
    main()