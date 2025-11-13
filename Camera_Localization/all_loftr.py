#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 16:27:57 2025

@author: nitaishah
"""

import re
import csv
from pathlib import Path
from copy import deepcopy
import cv2
import numpy as np
import torch
import sys

# === Path to LoFTR code ===
essential_path = Path("/Volumes/One Touch/COLMAP_RELLIS/essential/LoFTR-master")
sys.path.append(str(essential_path))

from src.loftr import LoFTR, default_cfg

# ===================== USER SETTINGS =====================
query_image  = Path("/Volumes/One Touch/Highway_Exp/OUTPUTS/frame_0.jpg")  
db_dir       = Path("/Volumes/One Touch/Highway_Exp/DATASETS/images")  
weights_ckpt = Path("/Volumes/One Touch/COLMAP_RELLIS/essential/LoFTR-master/weights/outdoor_ds.ckpt")

out_csv      = Path("/Volumes/One Touch/Highway_Exp/OUTPUTS/loftr_matches_all.csv")  
resize_dim   = (640, 480)   # (W,H)
conf_thresh  = 0.80         # confidence threshold
round_pixels = False        # round pixel coords in output
# ==========================================================

# ----------------- DEVICE -----------------
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ----------------- MODEL LOADING -----------------
def load_loftr(weights_path: Path):
    cfg = deepcopy(default_cfg)
    cfg['coarse']['temp_bug_fix'] = False
    matcher = LoFTR(config=cfg)
    state = torch.load(str(weights_path), map_location=device)
    matcher.load_state_dict(state['state_dict'])
    return matcher.eval().to(device)

# ----------------- IO & PREP -----------------
def imread_gray(path: Path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(f"Could not read {path}")
    return g

def prep_tensor(gray, target_wh):
    Wt, Ht = target_wh
    gray_res = cv2.resize(gray, (Wt, Ht), interpolation=cv2.INTER_AREA)
    ten = torch.from_numpy(gray_res)[None, None].float() / 255.
    return gray_res, ten.to(device)

def list_all_images(folder: Path):
    exts = ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG")
    paths = []
    for ext in exts:
        paths += list(folder.glob(ext))
    return sorted([p for p in paths if not p.name.startswith("._")])

# ----------------- MATCHING CORE -----------------
@torch.no_grad()
def loftr_match_q_vs_db(matcher, q_gray, q_tensor, q_size, db_path: Path, resize_wh=(640,480)):
    dg = imread_gray(db_path)
    Hq0, Wq0 = q_size
    Hd0, Wd0 = dg.shape[:2]
    Wt, Ht   = resize_wh

    dg_res, d_t = prep_tensor(dg, (Wt, Ht))

    batch = {'image0': q_tensor, 'image1': d_t}
    matcher(batch)

    mk0 = batch['mkpts0_f'].detach().cpu().numpy()
    mk1 = batch['mkpts1_f'].detach().cpu().numpy()
    mcf = batch['mconf'].detach().cpu().numpy()

    if mk0.size == 0:
        return np.empty((0,2)), np.empty((0,2)), np.empty((0,))

    sx_q, sy_q = Wt / Wq0, Ht / Hq0
    sx_d, sy_d = Wt / Wd0, Ht / Hd0

    qx = mk0[:, 0] / sx_q
    qy = mk0[:, 1] / sy_q
    dx = mk1[:, 0] / sx_d
    dy = mk1[:, 1] / sy_d

    return np.column_stack([qx, qy]), np.column_stack([dx, dy]), mcf

# ----------------- CSV HELPERS -----------------
def ensure_csv_header(csv_path: Path):
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["query_image","db_image","qx","qy","dx","dy","confidence"])

def append_matches(csv_path: Path, qname: str, dname: str,
                   qpts: np.ndarray, dpts: np.ndarray, conf: np.ndarray):
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        for (qx,qy), (dx,dy), c in zip(qpts, dpts, conf):
            if round_pixels:
                qx, qy = int(round(qx)), int(round(qy))
                dx, dy = int(round(dx)), int(round(dy))
                w.writerow([qname, dname, qx, qy, dx, dy, f"{float(c):.6f}"])
            else:
                w.writerow([qname, dname, f"{qx:.6f}", f"{qy:.6f}", f"{dx:.6f}", f"{dy:.6f}", f"{float(c):.6f}"])

# ----------------- MAIN -----------------
def main():
    if not query_image.exists():
        raise FileNotFoundError(f"Query image not found: {query_image}")
    if not db_dir.exists():
        raise FileNotFoundError(f"DB dir not found: {db_dir}")
    if not weights_ckpt.exists():
        raise FileNotFoundError(f"Weights not found: {weights_ckpt}")

    print(f"Device: {device}")
    matcher = load_loftr(weights_ckpt)
    print("✅ LoFTR model loaded.")

    # Prepare query
    qg = imread_gray(query_image)
    Hq0, Wq0 = qg.shape[:2]
    qg_res, q_t = prep_tensor(qg, resize_dim)

    # List all DB images (no ID filtering)
    db_paths = list_all_images(db_dir)
    if not db_paths:
        raise RuntimeError(f"No database images found under {db_dir}")

    ensure_csv_header(out_csv)

    total_raw = total_kept = 0

    for i, dp in enumerate(db_paths, 1):
        try:
            qpts, dpts, conf = loftr_match_q_vs_db(matcher, qg, q_t, (Hq0, Wq0), dp, resize_dim)
        except Exception as e:
            print(f"[{i}/{len(db_paths)}] {dp.name}: ERROR {e}")
            continue

        if conf.size == 0:
            print(f"[{i}/{len(db_paths)}] {dp.name}: 0 matches")
            continue

        mask = conf >= conf_thresh
        qpts_f = qpts[mask]
        dpts_f = dpts[mask]
        conf_f = conf[mask]

        append_matches(out_csv, query_image.name, dp.name, qpts_f, dpts_f, conf_f)

        total_raw  += conf.size
        total_kept += conf_f.size
        print(f"[{i}/{len(db_paths)}] {dp.name}: matches={conf.size} kept@>={conf_thresh}={conf_f.size}")

    print(f"\n✅ Done. Total raw matches: {total_raw} | kept: {total_kept}")
    print(f"📄 CSV saved to: {out_csv.resolve()}")

if __name__ == "__main__":
    main()
