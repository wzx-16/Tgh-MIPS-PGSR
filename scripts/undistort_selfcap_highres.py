#!/usr/bin/env python3
import argparse
import glob
import json
import re
from pathlib import Path

import cv2
import numpy as np


def load_calibration(calib_json_path: str):
    """Load either of these calibration JSON formats:

    1) New format, keyed by camera id / serial number:
       {
         "22139906": {"K": [[...]], "distCoeff": [[...]], "imgSize": [2048, 1500]}
       }

    2) Old format:
       {"calibrations": [{"cameraSN": "cam_001", "fx": ..., "k1": ..., ...}]}
    """
    with open(calib_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    calib_map = {}

    # Old format from the original script.
    if isinstance(data, dict) and "calibrations" in data:
        for item in data["calibrations"]:
            cam_id = str(item["cameraSN"])
            K = np.array(
                [
                    [item["fx"], 0.0, item["cx"]],
                    [0.0, item["fy"], item["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            D = np.array([item["k1"], item["k2"], item["p1"], item["p2"], item["k3"]], dtype=np.float64)
            w = int(item["imageSize"]["w"])
            h = int(item["imageSize"]["h"])
            alpha = float(item.get("rectifyAlpha", 0.0))
            calib_map[cam_id] = {"K": K, "D": D, "size": (w, h), "alpha": alpha}
        return calib_map

    # New uploaded format: top-level keys are camera IDs / serial numbers.
    if not isinstance(data, dict):
        raise ValueError("Unsupported calibration JSON: expected a top-level object/dict.")

    for cam_id, item in data.items():
        if not isinstance(item, dict):
            continue
        if "K" not in item or "distCoeff" not in item:
            continue

        K = np.asarray(item["K"], dtype=np.float64)
        D = np.asarray(item["distCoeff"], dtype=np.float64).reshape(-1)
        K = K.reshape((3, 3))
        if K.shape != (3, 3):
            raise ValueError(f"Camera {cam_id}: K must be 3x3, got {K.shape}")
        if D.size < 4:
            raise ValueError(f"Camera {cam_id}: distCoeff must have at least 4 values, got {D.size}")

        img_size = item.get("imgSize") or item.get("imageSize")
        if isinstance(img_size, dict):
            w, h = int(img_size["w"]), int(img_size["h"])
        elif isinstance(img_size, (list, tuple)) and len(img_size) >= 2:
            w, h = int(img_size[0]), int(img_size[1])
        else:
            w, h = None, None

        alpha = float(item.get("rectifyAlpha", 0.0))
        calib_map[str(cam_id)] = {"K": K, "D": D, "size": (w, h), "alpha": alpha}

    if not calib_map:
        raise ValueError("No valid camera calibrations found in JSON.")

    return calib_map


def read_cam_index_map(path: str | None):
    """Optional JSON mapping for filenames like cam01_00001.jpg.

    Example:
      {"cam01": "22139906", "cam02": "22139911"}
    or:
      {"1": "22139906", "2": "22139911"}
    """
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k).lower(): str(v) for k, v in raw.items()}


def filename_to_camera_id(filename: str, calib_map: dict, cam_index_map: dict | None = None) -> str:
    """Extract camera id from filenames such as:

    - 22139906_00001.jpg      -> 22139906
    - cam01_00001.jpg         -> uses cam_index_map if provided, else cam_001/first sorted camera fallback
    - cam1_00001.jpg          -> uses cam_index_map if provided, else cam_001/first sorted camera fallback

    The expected general form is: camid_frameid.jpg/png/jpeg
    """
    stem = Path(filename).stem
    if "_" not in stem:
        raise ValueError(f"Unsupported filename format: {filename}; expected camid_frameid.ext")

    cam_token = stem.split("_", 1)[0]
    cam_token_lower = cam_token.lower()
    cam_index_map = cam_index_map or {}

    # Direct match: best for filenames like 22139906_00001.jpg.
    if cam_token in calib_map:
        return cam_token

    # Explicit map match, e.g. cam01 -> 22139906.
    if cam_token_lower in cam_index_map:
        return cam_index_map[cam_token_lower]

    # Explicit map match by numeric part, e.g. cam01 -> key "1".
    m = re.fullmatch(r"cam0*(\d+)", cam_token_lower)
    if m:
        idx = int(m.group(1))
        if str(idx) in cam_index_map:
            return cam_index_map[str(idx)]

        old_style = f"cam_{idx:03d}"
        if old_style in calib_map:
            return old_style

        # Fallback: map cam1/cam01 to the first calibration key in sorted order.
        # This is only used when there is no explicit map and no cam_001-style key.
        sorted_ids = sorted(calib_map.keys())
        if 1 <= idx <= len(sorted_ids):
            return sorted_ids[idx - 1]

    raise ValueError(
        f"No calibration key found for {filename}. Extracted camera token '{cam_token}'. "
        "Use serial-number filenames or pass --cam_index_map."
    )


def undistort_one_image(img, K, D, alpha=0.0):
    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
    undist = cv2.undistort(img, K, D, None, new_K)
    return undist, roi


def main():
    parser = argparse.ArgumentParser(description="Undistort images named camid_frameid.jpg using a calibration JSON.")
    parser.add_argument("--input_dir", default="images", help="Folder containing distorted source images. Default: images")
    parser.add_argument("--output_dir", default="images_undistorted", help="Folder for undistorted images. Default: images_undistorted")
    parser.add_argument("--calib_json", default="calibration_new_with_k.json", help="Calibration JSON path.")
    parser.add_argument("--cam_index_map", default=None, help="Optional JSON map for cam01/cam1 filenames to calibration serials.")
    parser.add_argument("--alpha", type=float, default=None, help="Override rectification alpha. If omitted, uses per-camera rectifyAlpha or 0.0.")
    parser.add_argument("--crop_roi", action="store_true", help="Crop to OpenCV valid ROI after undistortion.")
    parser.add_argument("--overwrite", action="store_true", help="Allow output_dir to be the same as input_dir and overwrite files.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if input_dir.resolve() == output_dir.resolve() and not args.overwrite:
        raise ValueError("output_dir is the same as input_dir. Use --overwrite if you really want to replace originals.")

    output_dir.mkdir(parents=True, exist_ok=True)

    calib_map = load_calibration(args.calib_json)
    cam_index_map = read_cam_index_map(args.cam_index_map)

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        image_paths.extend(glob.glob(str(input_dir / ext)))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"No images found in: {input_dir}")
        return

    num_ok = 0
    num_skip = 0

    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        filename = img_path.name

        try:
            cam_id = filename_to_camera_id(filename, calib_map, cam_index_map)
        except ValueError as e:
            print(f"[SKIP] {e}")
            num_skip += 1
            continue

        if cam_id not in calib_map:
            print(f"[SKIP] Mapped {filename} to {cam_id}, but that key is not in calibration JSON")
            num_skip += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[SKIP] Failed to read {img_path}")
            num_skip += 1
            continue

        H, W = img.shape[:2]
        calib_W, calib_H = calib_map[cam_id]["size"]
        if calib_W is not None and calib_H is not None and (W, H) != (calib_W, calib_H):
            print(f"[WARN] {filename}: image size {(W, H)} != calibration size {(calib_W, calib_H)}")

        K = calib_map[cam_id]["K"]
        D = calib_map[cam_id]["D"]
        alpha = args.alpha if args.alpha is not None else calib_map[cam_id].get("alpha", 0.0)

        undist, roi = undistort_one_image(img, K, D, alpha=alpha)

        if args.crop_roi:
            x, y, w, h = roi
            if w > 0 and h > 0:
                undist = undist[150 : 2850, 152 : 3944]

        out_path = output_dir / filename
        ok = cv2.imwrite(str(out_path), undist)
        if not ok:
            print(f"[SKIP] Failed to write {out_path}")
            num_skip += 1
            continue

        print(f"[OK] {filename} -> {out_path} ({cam_id})")
        num_ok += 1

    print(f"\nDone. saved={num_ok}, skipped={num_skip}")


if __name__ == "__main__":
    main()