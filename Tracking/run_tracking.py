#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 18:59:42 2025

@author: nitaishah
"""

import supervision as sv
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
import cv2


# ----- SET DEVICE FOR HPRC (CUDA) -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----- SET PATHS (UPDATE FOR EACH VIDEO) -----
SOURCE_VIDEO_PATH = "/scratch/user/nitaishah/SH_47/TRACKING/INPUT/DJI_20250807102624_0570_D_001.MP4"  # e.g., "/scratch/user/nitaishah/traffic_videos/game1.mp4"
OUTPUT_VIDEO_PATH = "/scratch/user/nitaishah/SH_47/TRACKING/OUTPUTS/VIDEOS/DJI_20250807102624_0570_D_001_output.MP4" # e.g., "/scratch/user/nitaishah/processed_videos/game1_annotated.mp4"
CSV_OUTPUT_PATH =  "/scratch/user/nitaishah/SH_47/TRACKING/OUTPUTS/CSV/DJI_20250807102624_0570_D_001_output.csv"    # e.g., "/scratch/user/nitaishah/tracking_csvs/game1_tracking.csv"

# ----- LOAD YOLOv8 MODEL -----
model = YOLO("/scratch/user/nitaishah/traffic_count/yolov8x.pt")
CLASS_NAMES_DICT = model.model.names

# ----- SELECT VEHICLE CLASSES -----
SELECTED_CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
    for class_name in SELECTED_CLASS_NAMES
]

# ----- INITIALIZE BYTE TRACKER -----
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.2,
    lost_track_buffer=100,
    minimum_matching_threshold=0.5,
    frame_rate=30,
    minimum_consecutive_frames=3
)
byte_tracker.reset()

# ----- VIDEO INFO -----
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# Annotators (optional visualization)
box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.5, text_color=sv.Color.BLACK)
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)

# Data storage
tracking_data = []

# ----- CALLBACK FUNCTION FOR PROCESSING EACH FRAME -----
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    global tracking_data

    # Run YOLOv8 detection
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Filter for selected vehicle classes
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]

    # Track with ByteTrack
    detections = byte_tracker.update_with_detections(detections)

    # Record tracking info per vehicle
    for tracker_id, (x1, y1, x2, y2) in zip(detections.tracker_id, detections.xyxy):
        tracking_data.append({
            "Frame": index,
            "Vehicle_ID": tracker_id,
            "X": float((x1 + x2) / 2),
            "Y": float((y1 + y2) / 2),
            "Width": float(x2 - x1),
            "Height": float(y2 - y1)
        })

    # Annotate frame (optional visualization)
    labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    return annotated_frame

# ----- PROCESS ENTIRE VIDEO -----
sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=OUTPUT_VIDEO_PATH,
    callback=callback
)

print(f"Video processing complete. Output video saved at: {OUTPUT_VIDEO_PATH}")

# ----- SAVE TRACKING DATA TO CSV -----
df = pd.DataFrame(tracking_data)
df.to_csv(CSV_OUTPUT_PATH, index=False)
print(f" Tracking CSV saved at: {CSV_OUTPUT_PATH}")