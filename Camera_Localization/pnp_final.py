#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 01:28:50 2025

@author: nitaishah
"""

import os, tempfile, csv, cv2, torch, open3d as o3d
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from hloc.utils.read_write_model import read_points3D_binary

# ======================================================
# 🔧 Safety fix for PyTorch temp debug directory
# ======================================================
safe_debug_dir = os.path.join(tempfile.gettempdir(), "torch_compile_debug")
os.makedirs(safe_debug_dir, exist_ok=True)
os.environ["TORCH_COMPILE_DEBUG_DIR"] = safe_debug_dir

# ======================================================
# USER CONFIGURATION
# ======================================================
LOFTR_MATCHES_CSV = Path("/Users/nitaishah/Downloads/loftr_matches_all_20.csv")
REPROJ_CSV        = Path("/Volumes/One Touch/Highway_Exp/OUTPUTS/all_reproj_under_1.0px.csv")
MODEL_PATH        = Path("/Volumes/One Touch/Highway_Exp/DATASETS/sparse/1")
QUERY_IMAGE       = Path("/Volumes/One Touch/Highway_Exp/OUTPUTS/frame_0.jpg")

K_QUERY = np.array([[1391.38942, 0.0,        1836.97640],
                    [0.0,        1386.11820, 1365.85714],
                    [0.0,        0.0,        1.0]], dtype=np.float64)

CONF_MIN     = 0.20
RADIUS_PX    = 20.0
ERR_GUARD    = 5.0
RANSAC_ERR   = 10.0
RANSAC_CONF  = 0.999
IMG_W, IMG_H = 3840, 2160

# ======================================================
# HELPERS
# ======================================================

def load_reproj_grouped(path):
    by_name = {}
    with open(path, "r") as f:
        for r in csv.DictReader(f):
            nm = r.get("image_name", "")
            if not nm:
                continue
            by_name.setdefault(nm, []).append(r)

    per_img = {}
    for name, rows in by_name.items():
        xy, pid, err = [], [], []
        for r in rows:
            try:
                xy.append([float(r["kp_x"]), float(r["kp_y"])])
                pid.append(int(r["point3D_id"]))
                err.append(float(r["error_px"]))
            except:
                continue
        if xy:
            xy = np.asarray(xy, float)
            pid = np.asarray(pid, int)
            err = np.asarray(err, float)
            per_img[name] = {"xy": xy, "pid": pid, "err": err, "kdt": cKDTree(xy)}
    return per_img

def associate_q2d_x3d(qpts, dpts, conf, db_name, reproj_by_db, points3D):
    if db_name not in reproj_by_db:
        return np.empty((0,2)), np.empty((0,3))
    pack = reproj_by_db[db_name]
    tree, r_xy, r_pid, r_err = pack["kdt"], pack["xy"], pack["pid"], pack["err"]
    Q2D, X3D = [], []
    for (qx, qy), (dx, dy), c in zip(qpts, dpts, conf):
        if c < CONF_MIN: 
            continue
        idxs = tree.query_ball_point([dx, dy], r=RADIUS_PX)
        if not idxs: 
            continue
        j = min(idxs, key=lambda k: (r_xy[k,0]-dx)**2 + (r_xy[k,1]-dy)**2)
        if r_err[j] > ERR_GUARD: 
            continue
        pid = int(r_pid[j])
        if pid not in points3D:
            continue
        Q2D.append([qx, qy])
        X3D.append(points3D[pid].xyz)
    return np.asarray(Q2D, float), np.asarray(X3D, float)

def load_colmap_pointcloud(points3d_path):
    pts = read_points3D_binary(points3d_path)
    xyz, rgb = [], []
    for p in pts.values():
        xyz.append(p.xyz.astype(np.float64))
        c = getattr(p, "rgb", np.array([200,200,200], dtype=np.uint8))
        rgb.append(c.astype(np.float64)/255.0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(rgb))
    return pcd

def Rt_to_T(R, t):
    T = np.eye(4)
    T[:3,:3] = R.T
    T[:3,3] = (-R.T @ t).ravel()
    return T

def make_camera_frustum(K, img_size, depth=1.0, T_cw=None, color=(1,0,0)):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    W, H = img_size
    corners_px = np.array([[0,0],[W,0],[W,H],[0,H]], float)
    pts_cam = np.vstack([[0,0,0], [[(u-cx)/fx*depth, (v-cy)/fy*depth, depth] for u,v in corners_px]])
    if T_cw is not None:
        pts_h = np.hstack([pts_cam, np.ones((5,1))])
        pts_cam = (T_cw @ pts_h.T).T[:, :3]
    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_cam)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color]*len(lines))
    return ls

# ======================================================
# MAIN PIPELINE
# ======================================================
def normalize_name(name):
    """Normalize filenames to lowercase and remove extensions."""
    return str(name).lower().replace(".jpg", "").replace(".png", "")

def localize_from_csv(selected_ids, selected_yaws, visualize=True):
    df = pd.read_csv(LOFTR_MATCHES_CSV)
    df["db_norm"] = df["db_image"].map(normalize_name)

    reproj_by_db = load_reproj_grouped(REPROJ_CSV)
    # Normalize reprojection dict keys too
    reproj_by_db = {normalize_name(k): v for k, v in reproj_by_db.items()}

    points3D = read_points3D_binary(MODEL_PATH / "points3D.bin")

    # Filter by confidence
    df = df[df["confidence"] >= CONF_MIN]

    # Extract normalized ID and yaw
    df["_id"]  = df["db_norm"].str.split("_").str[0]
    df["_yaw"] = df["db_norm"].str.extract(r"yaw(\d+)")
    df = df[df["_id"].isin(selected_ids) & df["_yaw"].isin([str(y) for y in selected_yaws])]
    if df.empty:
        print("❌ No matches found for given IDs/Yaws.")
        return

    print(f"Loaded {len(df)} matches across {df['db_image'].nunique()} images.")

    all_Q2D, all_X3D = [], []
    for db_name, g in df.groupby("db_norm"):
        qpts = g[["qx","qy"]].to_numpy(float)
        dpts = g[["dx","dy"]].to_numpy(float)
        conf = g["confidence"].to_numpy(float)
        Q2D, X3D = associate_q2d_x3d(qpts, dpts, conf, db_name, reproj_by_db, points3D)
        print(f"{db_name}: {len(Q2D)} valid 2D–3D correspondences")
        if len(Q2D) > 0:
            all_Q2D.append(Q2D)
            all_X3D.append(X3D)

    if not all_Q2D:
        print("❌ No valid 2D–3D pairs found.")
        return

    Q2D = np.vstack(all_Q2D)
    X3D = np.vstack(all_X3D)
    print(f"\n✅ Total correspondences: {len(Q2D)}")

    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        X3D, Q2D, K_QUERY, None,
        reprojectionError=RANSAC_ERR,
        confidence=RANSAC_CONF,
        flags=cv2.SOLVEPNP_EPNP
    )
    if not ok:
        print("❌ PnP failed.")
        return

    R, _ = cv2.Rodrigues(rvec)
    C = -R.T @ tvec
    print("\n✅ PnP Success")
    print("Camera Center:", C.ravel())
    print("Rotation:\n", R)
    print("Translation:\n", tvec.ravel())

    if visualize:
        pcd = load_colmap_pointcloud(MODEL_PATH / "points3D.bin")
        T_cw = Rt_to_T(R, tvec)
        frustum = make_camera_frustum(K_QUERY, (IMG_W, IMG_H), 1.0, T_cw, color=(1,0,0))
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        cam_frame.transform(T_cw)
        o3d.visualization.draw_geometries([pcd, frustum, cam_frame],
                                          window_name=f"Pose from IDs {selected_ids}, yaws {selected_yaws}",
                                          width=1280, height=800)


if __name__ == "__main__":
    selected_ids  = ["000000"]
    selected_yaws = [120, 150]   # You can change this freely
    localize_from_csv(selected_ids, selected_yaws, visualize=True)
