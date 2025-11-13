#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 02:36:55 2025

@author: nitaishah
"""

import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import csv
import re
from hloc.utils.read_write_model import read_model  # ensure this file is accessible

# === CONFIG ===
model_path = Path("/Volumes/One Touch/Highway_Exp/DATASETS/sparse/1/")  # cameras.bin, images.bin, points3D.bin
image_path = Path("/Volumes/One Touch/Highway_Exp/DATASETS/images")
err_thresh = 1.0  # reprojection error threshold (px)
output_csv = f"all_reproj_under_{err_thresh:.1f}px.csv"

# === Load COLMAP model ===
print("Loading COLMAP model...")
cameras, images, points3D = read_model(model_path, ext=".bin")
print(f"Loaded {len(cameras)} cameras, {len(images)} images, {len(points3D)} points3D")

# === Build fast lookup ===
name_to_image = {im.name: im for im in images.values()}
colmap_names = set(name_to_image.keys())

# === Regex to extract numeric prefix (e.g. 000000_...) ===
id_pattern = re.compile(r"^(\d+)_")

# --- Find matching images ---
matching_images = []
for img_file in image_path.glob("*.jpg"):
    fname = img_file.name
    if fname.startswith("._"):  # skip hidden macOS files
        continue
    # Only process if this file exists in the COLMAP model
    if fname in colmap_names:
        matching_images.append(fname)

matching_images = sorted(set(matching_images))
if not matching_images:
    raise ValueError("❌ No matching images found in both the folder and COLMAP model.")

print(f"✅ Found {len(matching_images)} valid images shared between folder and COLMAP model.")

# === Process and save reprojection results ===
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "point3D_id", "kp_x", "kp_y", "proj_u", "proj_v", "error_px"])

    for image_name in matching_images:
        print(f"\nProcessing {image_name}...")
        image = name_to_image[image_name]
        camera = cameras[image.camera_id]

        # Intrinsics (SIMPLE_RADIAL or PINHOLE)
        if len(camera.params) == 4:  # SIMPLE_RADIAL
            fx, cx, cy, _ = camera.params
            K = np.array([[fx, 0, cx],
                          [0, fx, cy],
                          [0, 0, 1]])
        elif len(camera.params) == 5:  # PINHOLE
            fx, fy, cx, cy, _ = camera.params
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
        else:
            print(f"[WARN] Unsupported camera model for {image_name}; skipping.")
            continue

        # Pose
        R = image.qvec2rotmat()
        t = image.tvec.reshape(3, 1)
        P = K @ np.hstack((R, t))

        # Load image to confirm presence
        img = cv2.imread(str(image_path / image_name))
        if img is None:
            print(f"[WARN] Cannot read {image_name}; skipping.")
            continue

        # Compute reprojection errors
        matches_under_thresh = []
        for kp, pid in zip(image.xys, image.point3D_ids):
            if pid == -1 or pid not in points3D:
                continue
            X = np.append(points3D[pid].xyz, 1.0)
            x_proj = P @ X
            x_proj /= x_proj[2]
            err = float(np.hypot(kp[0] - x_proj[0], kp[1] - x_proj[1]))
            if err < err_thresh:
                matches_under_thresh.append((kp, x_proj, err, pid))

        print(f"Total valid 2D–3D matches below {err_thresh}px: {len(matches_under_thresh)}")

        # Write all matches to CSV
        for (kp_xy, proj_uv, e, pid) in matches_under_thresh:
            writer.writerow([
                image_name,
                pid,
                f"{kp_xy[0]:.6f}", f"{kp_xy[1]:.6f}",
                f"{proj_uv[0]:.6f}", f"{proj_uv[1]:.6f}",
                f"{e:.6f}"
            ])

print(f"\n✅ All results saved to: {output_csv}")
