#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 16:02:06 2025

@author: nitaishah
"""

import numpy as np
import pandas as pd
import open3d as o3d
from pathlib import Path

# ========================= USER INPUTS =========================
PLY_IN  = Path("/Users/nitaishah/Desktop/SH_47/Local_debug/fused.ply")
BBOXES_CSV = Path("/Volumes/One Touch/Highway_Exp/OUTPUTS/DJI_20250807102624_0570_D_001_output.csv")

# --- Camera intrinsics ---
K = np.array([[1391.38942, 0.0,        1836.97640],
              [0.0,        1386.11820, 1365.85714],
              [0.0,        0.0,        1.0      ]], dtype=np.float64)

IMG_WIDTH, IMG_HEIGHT = 3840, 2160
RESCALE_INTRINSICS_TO_IMAGE = True
K_REF_SIZE = (2944, 2061)

# --- Camera extrinsics (world → camera) ---
R = np.array([
    [0.8508306,   0.18354079,  0.49234141],
    [-0.30186447,  0.93768854,  0.17209894],
    [-0.43007572, -0.29504742,  0.85321855]
])

# Translation vector (as a column vector)
t = np.array([[-2.28826885],
              [-0.71433363],
              [ 5.37345421]])

# --- Cloud scaling and cleaning ---
METRIC_SCALE = 8.19        # <--- set this to your known scale factor
SCALE_T_WITH_WORLD = True    # scale translation vector as well
VOXEL_SIZE = 0.03 * METRIC_SCALE
USE_SOR = True;  SOR_NN = 20;  SOR_STD = 2.0
USE_ROR = True;  ROR_MIN_PTS = 16;  ROR_RADIUS = 0.10 * METRIC_SCALE

# --- Ground plane ---
FIT_GROUND_FROM_PC = True
PLANE_N = np.array([0,1,0]); PLANE_D = 0.0

# --- Visualization ---
DRAW_FRUSTUM = True
FRUSTUM_DEPTH = 1.0
POINT_RADIUS = 0.05
WINDOW_W, WINDOW_H = 1280, 800

# --- Corner colors (bottom corners only) ---
CORNER_COLORS = {1: (1, 0, 0), 2: (0, 1, 0)}  # bottom-left = red, bottom-right = green
# =============================================================


# ----------------------- Helper functions -----------------------
def summarize(tag, pcd):
    print(f"{tag}: {len(pcd.points):,} pts")

def rescale_intrinsics(K_in, ref_size, new_size):
    w_ref, h_ref = ref_size
    w_new, h_new = new_size
    sx, sy = w_new / w_ref, h_new / h_ref
    K_out = K_in.copy()
    K_out[0,0] *= sx; K_out[1,1] *= sy
    K_out[0,2] *= sx; K_out[1,2] *= sy
    return K_out

def load_pointcloud_ply(ply_path: Path):
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise FileNotFoundError(f"Empty PLY: {ply_path}")
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.8, 0.8, 0.8])
    return pcd

def apply_uniform_scale(pcd, s, scale_t, t_vec):
    if abs(s - 1.0) < 1e-12: return pcd, t_vec
    pcd.scale(s, center=(0,0,0))
    if scale_t: t_vec = s * t_vec
    return pcd, t_vec

def fit_ground_plane_o3d(pcd, dist_thresh=0.05):
    plane, inliers = pcd.segment_plane(distance_threshold=dist_thresh, ransac_n=3, num_iterations=3000)
    a,b,c,d = plane
    n = np.array([a,b,c]); n /= np.linalg.norm(n)
    d /= np.linalg.norm([a,b,c])
    return n, d

def cam_center_world(R, t): 
    return (-R.T @ t).ravel()

def pixel_from_bbox_corner(x, y, w, h, corner_id):
    if corner_id == 1:   # bottom-left
        u, v = x, y + h
    elif corner_id == 2: # bottom-right
        u, v = x + w, y + h
    else:
        raise ValueError("Only bottom corners (1 or 2) allowed.")
    return float(u), float(v)

def pixel_ray_dir_in_camera(Kp, u, v):
    fx, fy, cx, cy = Kp[0,0], Kp[1,1], Kp[0,2], Kp[1,2]
    x, y = (u - cx) / fx, (v - cy) / fy
    d = np.array([x, y, 1.0])
    return d / np.linalg.norm(d)

def ray_plane_intersection(C, d_cam, Rcw, plane_n, plane_d):
    d_world = (Rcw.T @ d_cam.reshape(3,1)).ravel()
    denom = plane_n.dot(d_world)
    if abs(denom) < 1e-10: return None
    lam = -(plane_n.dot(C) + plane_d) / denom
    return None if lam < 0 else C + lam * d_world

def compute_corner_trajectory(track, Kp, Rcw, tcw, plane_n, plane_d, img_w, img_h, corner_id):
    C = cam_center_world(Rcw, tcw)
    pts = []
    for (x, y, w, h) in track:
        u, v = pixel_from_bbox_corner(x, y, w, h, corner_id)
        u, v = np.clip(u, 0, img_w-1), np.clip(v, 0, img_h-1)
        d_cam = pixel_ray_dir_in_camera(Kp, u, v)
        Xw = ray_plane_intersection(C, d_cam, Rcw, plane_n, plane_d)
        if Xw is not None:
            pts.append(Xw)
    return np.array(pts)

def traj_geometry(points_xyz, color=(1,0,0), vertex_radius=0.03):
    geoms = []
    pts = np.asarray(points_xyz)
    if len(pts) >= 2:
        lines = [[i, i+1] for i in range(len(pts)-1)]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines  = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([color]*len(lines))
        geoms.append(ls)
    for p in pts:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=vertex_radius)
        s.translate(p)
        s.paint_uniform_color(color)
        geoms.append(s)
    return geoms

def make_camera_frustum(Kp, img_size, depth=1.0, T_cw=None, color=(1,0,0)):
    fx, fy, cx, cy = Kp[0,0], Kp[1,1], Kp[0,2], Kp[1,2]
    W, H = img_size
    corners = np.array([[0,0],[W,0],[W,H],[0,H]])
    pts_cam = [[0,0,0]]
    for (u,v) in corners:
        x, y, z = (u - cx)/fx * depth, (v - cy)/fy * depth, depth
        pts_cam.append([x,y,z])
    pts_cam = np.asarray(pts_cam)
    if T_cw is not None:
        pts_h = np.hstack([pts_cam, np.ones((5,1))])
        pts_cam = (T_cw @ pts_h.T).T[:,:3]
    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_cam)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color]*len(lines))
    return ls

def trajectory_length(points_xyz):
    """Compute total 3D length of a trajectory in meters."""
    pts = np.asarray(points_xyz)
    if len(pts) < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return float(np.sum(dists))
# ----------------------------------------------------------------


# ---------------------------- MAIN ----------------------------
def main():
    # 1) Load dense cloud
    pcd = load_pointcloud_ply(PLY_IN)
    summarize("Loaded", pcd)

    # 2) Metric scaling
    tcw = t.copy()
    pcd, tcw = apply_uniform_scale(pcd, METRIC_SCALE, SCALE_T_WITH_WORLD, tcw)

    # 3) Clean
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    if USE_SOR:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=SOR_NN, std_ratio=SOR_STD)
    if USE_ROR:
        pcd, _ = pcd.remove_radius_outlier(nb_points=ROR_MIN_PTS, radius=ROR_RADIUS)
    summarize("Cleaned", pcd)

    # 4) Ground plane
    if FIT_GROUND_FROM_PC:
        plane_n, plane_d = fit_ground_plane_o3d(pcd)
        print(f"[plane] fitted n={plane_n}, d={plane_d:.3f}")
    else:
        plane_n, plane_d = PLANE_N, PLANE_D

    # 5) Adjust intrinsics if needed
    Kp = rescale_intrinsics(K, K_REF_SIZE, (IMG_WIDTH, IMG_HEIGHT)) if RESCALE_INTRINSICS_TO_IMAGE else K.copy()

    # 6) Load bounding boxes
    df = pd.read_csv(BBOXES_CSV)
    if "Vehicle_ID" not in df.columns:
        raise ValueError("CSV must contain a 'Vehicle_ID' column.")

    unique_ids = sorted(df["Vehicle_ID"].unique())
    print(f"Available Vehicle IDs: {unique_ids}")
    chosen = input("Enter Vehicle IDs (comma-separated, or press Enter for first): ").strip()
    if not chosen:
        chosen_ids = [unique_ids[0]]
    else:
        chosen_ids = [int(c.strip()) for c in chosen.split(",") if c.strip().isdigit()]

    geoms = [pcd]

    # 7) For each selected vehicle, visualize bottom corners
    for vid in chosen_ids:
        df_id = df[df["Vehicle_ID"] == vid].sort_values("Frame").reset_index(drop=True)
        print(f"\n[Vehicle {vid}] {len(df_id)} frames")

        track = df_id[["X","Y","Width","Height"]].to_numpy()

        for corner_id, color in CORNER_COLORS.items():
            traj = compute_corner_trajectory(track, Kp, R, tcw, plane_n, plane_d,
                                             IMG_WIDTH, IMG_HEIGHT, corner_id)
            length_m = trajectory_length(traj)
            print(f"  Corner {corner_id} → {len(traj)} 3D points, length = {length_m:.2f} m")
            geoms += traj_geometry(traj, color=color, vertex_radius=POINT_RADIUS)

    # 8) Add camera frustum
    if DRAW_FRUSTUM:
        T_cw = np.eye(4)
        T_cw[:3,:3] = R.T
        T_cw[:3,3] = cam_center_world(R, tcw)
        geoms.append(make_camera_frustum(Kp, (IMG_WIDTH, IMG_HEIGHT), depth=FRUSTUM_DEPTH, T_cw=T_cw))

    # 9) Visualize everything
    o3d.visualization.draw_geometries(
        geoms,
        window_name="3D Trajectories (Bottom Corners Only)",
        width=WINDOW_W,
        height=WINDOW_H
    )

if __name__ == "__main__":
    main()
