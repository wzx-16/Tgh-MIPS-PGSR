#!/usr/bin/env python3
import os
import re
import glob
import json
import argparse

import cv2
import numpy as np


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / np.linalg.norm(q)

    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)
    return R


def parse_image_filename(filename):
    """
    cam01_00001.jpg -> ("cam_001", 1)
    cam1_00001.jpg  -> ("cam_001", 1)
    """
    m = re.match(r"^cam(\d+)_(\d+)\.(jpg|jpeg|png)$", filename, re.IGNORECASE)
    if not m:
        return None, None
    cam_idx = int(m.group(1))
    frame_idx = int(m.group(2))
    cam_sn = f"cam_{cam_idx:03d}"
    return cam_sn, frame_idx


def load_calib_and_compute_demo_intrinsics(calib_path):
    with open(calib_path, "r") as f:
        calib = json.load(f)

    if "calibrations" not in calib:
        raise ValueError(f"Expected 'calibrations' in {calib_path}")

    cameras = {}

    for item in calib["calibrations"]:
        cam_sn = item["cameraSN"]

        fx = float(item["fx"])
        fy = float(item["fy"])
        cx = float(item["cx"])
        cy = float(item["cy"])

        W = int(item["imageSize"]["w"])
        H = int(item["imageSize"]["h"])

        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # same OpenCV distortion ordering used in your other script
        D = np.array([
            float(item.get("k1", 0.0)),
            float(item.get("k2", 0.0)),
            float(item.get("p1", 0.0)),
            float(item.get("p2", 0.0)),
            float(item.get("k3", 0.0)),
        ], dtype=np.float64)

        # world2Cam
        w2c_info = item["world2Cam"]
        R_w2c = quaternion_to_rotation_matrix(
            float(w2c_info["qw"]),
            float(w2c_info["qx"]),
            float(w2c_info["qy"]),
            float(w2c_info["qz"]),
        )
        t_w2c = np.array([
            float(w2c_info["x"]),
            float(w2c_info["y"]),
            float(w2c_info["z"]),
        ], dtype=np.float64)

        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c

        # demo script inverts this matrix
        c2w = np.linalg.inv(w2c)

        # ===== match demo intrinsic processing =====
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            K, D, (W, H), 1, (W, H)
        )

        # demo shifts principal point
        new_camera_matrix[0, 2] -= 48
        new_camera_matrix[1, 2] -= 36

        W2 = W - 96
        H2 = H - 72

        # demo scales by /3
        new_camera_matrix[0, 0] /= 3.0
        new_camera_matrix[1, 1] /= 3.0
        new_camera_matrix[0, 2] /= 3.0
        new_camera_matrix[1, 2] /= 3.0
        W2 = W2 // 3
        H2 = H2 // 3
        # ==========================================

        cameras[cam_sn] = {
            "K_demo": new_camera_matrix,
            "w": int(W2),
            "h": int(H2),
            "c2w": c2w,
        }

    return cameras


def convert_pose_like_demo(c2w):
    """
    Match the pose convention in your demo:
        poses[:, 0:3, 1] *= -1
        poses[:, 0:3, 2] *= -1
    """
    pose = c2w.copy()
    pose[:3, 1] *= -1
    pose[:3, 2] *= -1
    return pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".", help="Dataset root")
    parser.add_argument("--calib_json", default="calib.json", help="Calibration json")
    parser.add_argument("--images_dir", default="images", help="Image folder")
    # parser.add_argument("--output", default=".", help="Output transforms file")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS for time field")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    calib_path = os.path.join(data_dir, args.calib_json)
    images_dir = os.path.join(data_dir, args.images_dir)
    #output_path = os.path.join(data_dir, args.output)

    camera_dict = load_calib_and_compute_demo_intrinsics(calib_path)

    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    image_files = sorted(set(image_files))

    if not image_files:
        raise RuntimeError(f"No images found in {images_dir}")

    test_frames = []
    train_frames = []
    used_cameras = set()

    for img_path in image_files:
        filename = os.path.basename(img_path)
        cam_sn, frame_idx = parse_image_filename(filename)
        if frame_idx != 100:
            continue
        if cam_sn is None:
            print(f"[SKIP] unsupported filename format: {filename}")
            continue

        if cam_sn not in camera_dict:
            print(f"[SKIP] no calibration for {filename} -> {cam_sn}")
            continue

        cam = camera_dict[cam_sn]
        K_demo = cam["K_demo"]
        pose = convert_pose_like_demo(cam["c2w"])

        rel_path = os.path.relpath(img_path, data_dir)
        rel_path_no_ext = os.path.splitext(rel_path)[0].replace("\\", "/")

        frame = {
            "file_path": rel_path_no_ext,
            "fl_x": float(K_demo[0, 0]),
            "fl_y": float(K_demo[1, 1]),
            "cx": float(K_demo[0, 2]),
            "cy": float(K_demo[1, 2]),
            "transform_matrix": pose.tolist(),
            "time": frame_idx / args.fps,
        }
        if cam_sn == "cam_026":
            test_frames.append(frame)
        else:
            train_frames.append(frame)
        used_cameras.add(cam_sn)

    if not train_frames:
        raise RuntimeError("No valid frames matched between images and calibration")

    # like the demo/sample: top-level values come from one representative camera
    first_frame = train_frames[0]
    first_cam = camera_dict[parse_image_filename(os.path.basename(image_files[0]))[0]]

    train_transforms = {
        "w": int(first_cam["w"]),
        "h": int(first_cam["h"]),
        "fl_x": first_frame["fl_x"],
        "fl_y": first_frame["fl_y"],
        "cx": first_frame["cx"],
        "cy": first_frame["cy"],
        "frames": train_frames,
    }

    test_transforms = {
        "w": int(first_cam["w"]),
        "h": int(first_cam["h"]),
        "fl_x": first_frame["fl_x"],
        "fl_y": first_frame["fl_y"],
        "cx": first_frame["cx"],
        "cy": first_frame["cy"],
        "frames": test_frames,
    }

    train_output_path = os.path.join(args.data_dir, 'transforms_train.json')
    test_output_path = os.path.join(args.data_dir, 'transforms_test.json')
    print(f'[INFO] write to {train_output_path} and {test_output_path}')
    with open(train_output_path, 'w') as f:
        json.dump(train_transforms, f, indent=2)
    with open(test_output_path, 'w') as f:
        json.dump(test_transforms, f, indent=2)

    # with open(output_path, "w") as f:
    #     json.dump(transforms, f, indent=2)

    print(f"[INFO] loaded {len(camera_dict)} cameras from {calib_path}")
    print(f"[INFO] matched {len(used_cameras)} cameras and {len(train_frames) + len(test_frames)} images")
    print(f"[INFO] wrote {train_output_path} and {test_output_path}")


if __name__ == "__main__":
    main()