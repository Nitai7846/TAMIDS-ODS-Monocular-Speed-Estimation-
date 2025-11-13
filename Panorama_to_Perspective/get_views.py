#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 23:38:00 2025

@author: nitaishah
"""

import cv2
import py360convert
import os
from glob import glob

# === CONFIG ===
input_dir = "/scratch/user/nitaishah/SH_47/3D_RECONSTRUCTION/INPUT_PANOS/pano_data"
output_base = "/scratch/user/nitaishah/SH_47/3D_RECONSTRUCTION/PERSPECTIVE"

fov_deg = 90
pitch_angles = [-15]
yaw_step = 30
sky_crop_ratio = 0.3  # Crop top 30% (sky)

# === Create Output Base ===
os.makedirs(output_base, exist_ok=True)

# === Sky Crop Function ===
def remove_sky(image, ratio=sky_crop_ratio):
    h = image.shape[0]
    return image[int(h * ratio):, :]

# === Gather All Panoramas ===
image_paths = glob(os.path.join(input_dir, "*.png"))  # Adjust if using .jpeg or .png

for img_path in image_paths:
    eq_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if eq_img is None:
        print(f"Skipping unreadable image: {img_path}")
        continue

    eq_img = cv2.cvtColor(eq_img, cv2.COLOR_BGR2RGB)
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    # Use half panorama width for output view size
    pano_width = eq_img.shape[1]
    output_w = pano_width // 2
    output_size = (output_w, output_w)

    for pitch in pitch_angles:
        pitch_name = f"pitch{abs(pitch)}"
        pitch_dir = os.path.join(output_base, pitch_name)
        os.makedirs(pitch_dir, exist_ok=True)

        for yaw in range(0, 360, yaw_step):
            persp = py360convert.e2p(eq_img, fov_deg, yaw, pitch, output_size)
            persp_cropped = remove_sky(persp)

            filename = f"{img_name}_yaw{yaw}_pitch{pitch}.png"
            output_path = os.path.join(pitch_dir, filename)

            cv2.imwrite(output_path, cv2.cvtColor(persp_cropped, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(f"Saved: {output_path}")
