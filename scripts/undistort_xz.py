#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np


def build_calib_map(calib_json_path: str):
    with open(calib_json_path, "r") as f:
        data = json.load(f)

    if "calibrations" not in data:
        raise ValueError(
            f"Expected key 'calibrations' in {calib_json_path}, "
            "but it was not found."
        )

    calib_map = {}
    for item in data["calibrations"]:
        cam_sn = item["cameraSN"]  # e.g. cam_001

        K = np.array(
            [
                [item["fx"], 0.0, item["cx"]],
                [0.0, item["fy"], item["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # OpenCV distortion order: [k1, k2, p1, p2, k3]
        D = np.array(
            [item["k1"], item["k2"], item["p1"], item["p2"], item["k3"]],
            dtype=np.float64,
        )

        w = int(item["imageSize"]["w"])
        h = int(item["imageSize"]["h"])

        calib_map[cam_sn] = {
            "K": K,
            "D": D,
            "size": (w, h),
        }

    return calib_map


def filename_to_camera_sn(filename: str) -> str:
    """
    cam01_00001.jpg -> cam_001
    cam1_00001.jpg  -> cam_001
    """
    m = re.match(r"^cam(\d+)_\d+\.(jpg|jpeg|png)$", filename, re.IGNORECASE)
    if not m:
        raise ValueError(f"Unsupported filename format: {filename}")
    cam_idx = int(m.group(1))
    return f"cam_{cam_idx:03d}"


def undistort_one_image(img, K, D, alpha=0.0):
    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
    undist = cv2.undistort(img, K, D, None, new_K)
    return undist, roi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="images_origin",
        help="Folder containing source images, default: images_origin",
    )
    parser.add_argument(
        "--output_dir",
        default="images",
        help="Folder to save undistorted images, default: images",
    )
    parser.add_argument(
        "--calib_json",
        default="calib.json",
        help="Calibration json path, default: calib.json",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help=(
            "Free scaling parameter for getOptimalNewCameraMatrix. "
            "0.0 = less black border, 1.0 = keep more FOV. Default: 0.0"
        ),
    )
    parser.add_argument(
        "--crop_roi",
        action="store_true",
        help="Crop output to valid ROI returned by OpenCV",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calib_map = build_calib_map(args.calib_json)

    patterns = [
        str(input_dir / "cam??_*****.jpg"),
        str(input_dir / "cam??_*****.jpeg"),
        str(input_dir / "cam??_*****.png"),
        str(input_dir / "cam?_*****.jpg"),
        str(input_dir / "cam?_*****.jpeg"),
        str(input_dir / "cam?_*****.png"),
    ]

    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob.glob(pattern))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"No matching images found in: {input_dir}")
        return

    num_ok = 0
    num_skip = 0

    for img_path in image_paths:
        img_path = Path(img_path)
        filename = img_path.name

        try:
            cam_sn = filename_to_camera_sn(filename)
        except ValueError as e:
            print(f"[SKIP] {e}")
            num_skip += 1
            continue

        if cam_sn not in calib_map:
            print(f"[SKIP] No calibration found for {filename} -> {cam_sn}")
            num_skip += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[SKIP] Failed to read {img_path}")
            num_skip += 1
            continue

        H, W = img.shape[:2]
        calib_W, calib_H = calib_map[cam_sn]["size"]

        if (W, H) != (calib_W, calib_H):
            print(
                f"[WARN] {filename}: image size {(W, H)} != calib size {(calib_W, calib_H)}. "
                "Using actual image size for undistortion."
            )

        K = calib_map[cam_sn]["K"].copy()
        D = calib_map[cam_sn]["D"]

        undist, roi = undistort_one_image(img, K, D, alpha=args.alpha)

        if args.crop_roi:
            x, y, w, h = roi
            if w > 0 and h > 0:
                undist = undist[36:2736, 48:3648]

        out_path = output_dir / filename
        ok = cv2.imwrite(str(out_path), undist)
        if not ok:
            print(f"[SKIP] Failed to write {out_path}")
            num_skip += 1
            continue

        print(f"[OK] {filename} -> {out_path} ({cam_sn})")
        num_ok += 1

    print(f"\nDone. saved={num_ok}, skipped={num_skip}")


if __name__ == "__main__":
    main()