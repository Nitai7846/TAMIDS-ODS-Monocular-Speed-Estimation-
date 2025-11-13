"""
Function: Convert equirectangular image to perspective image
Reference: https://github.com/fuenwang/Equirec2Perspec/blob/master/Equirec2Perspec.py
"""

import os
import sys
import cv2
import numpy as np


def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out


def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out


class Equirectangular:
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape
        # cp = self._img.copy()
        # w = self._width
        # self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
        # self._img[:, w/8:, :] = cp[:, :7*w/8, :]

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ], np.float32)
        K_inv = np.linalg.inv(K)

        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        xyz = xyz @ R.T
        lonlat = xyz2lonlat(xyz)
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
        persp = cv2.remap(self._img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp
    

def convert_equirect_perspective(pano_dir, pano_name, output_dir):
    # Convert from equirectangular to perspective
    panoimg_path = os.path.join(pano_dir, pano_name + '.png')
    assert os.path.exists(panoimg_path)

    pano_output_dir = os.path.join(output_dir, pano_name)
    os.makedirs(pano_output_dir, exist_ok=True)

    equ = Equirectangular(panoimg_path)

    theta_range = np.arange(-180, 180, 45)  # yaw
    phi_range = np.array([0])  # pitch

    # CAMERA PARAMETERS
    FOV = 90
    H, W = 540, 960

    idx = 0
    for phi in phi_range:
        for theta in theta_range:
            # if idx % 30 == 0:
            #     print('{}/{}'.format(idx, len(theta_range) * len(phi_range)))
            img = equ.GetPerspective(FOV, theta, phi, H, W)  # Specify parameters(FOV, theta, phi, height, width)
            perspective_path = os.path.join(pano_output_dir, "{}_{:06d}.png".format(pano_name, idx))
            cv2.imwrite(perspective_path, img)
            idx += 1