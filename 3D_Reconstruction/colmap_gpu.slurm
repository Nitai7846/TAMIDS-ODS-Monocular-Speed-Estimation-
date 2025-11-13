#!/bin/bash

## NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=colmap_gpu_job             # Job name
#SBATCH --time=06:00:00                       # Wall time
#SBATCH --ntasks=1                            # One task
#SBATCH --mem=128G                            # Memory
#SBATCH --output=colmap_output.%j             # STDOUT log
#SBATCH --error=colmap_error.%j               # STDERR log
#SBATCH --gres=gpu:2                          # Request 2 GPUs
#SBATCH --partition=gpu                       # Use GPU partition

# Load GPU-enabled COLMAP module
ml purge
ml GCC/12.2.0 OpenMPI/4.1.4 COLMAP/3.11.1-CUDA-12.3.0

# === UPDATED PATHS ===
IMAGE_DIR="/scratch/user/nitaishah/3D_RECONSTRUCTION/images"
WORKSPACE="/scratch/user/nitaishah/3D_RECONSTRUCTION/output_backup_sh47"
DB_PATH="$WORKSPACE/database.db"
SPARSE_DIR="$WORKSPACE/sparse"
DENSE_DIR="$WORKSPACE/dense"

mkdir -p "$WORKSPACE" "$SPARSE_DIR" "$DENSE_DIR"

# === COLMAP PIPELINE ===

# Step 1: Feature Extraction (GPU)
colmap feature_extractor \
    --database_path "$DB_PATH" \
    --image_path "$IMAGE_DIR" \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --ImageReader.single_camera 0 \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.gpu_index 0 \
    --SiftExtraction.num_threads 8

# Step 2: Exhaustive Matching (GPU)
colmap exhaustive_matcher \
    --database_path "$DB_PATH" \
    --SiftMatching.use_gpu 1 \
    --SiftMatching.gpu_index 0 \
    --SiftMatching.num_threads 8

# Step 3: Sparse Reconstruction
colmap mapper \
    --database_path "$DB_PATH" \
    --image_path "$IMAGE_DIR" \
    --output_path "$SPARSE_DIR"

# Step 4: Image Undistortion
colmap image_undistorter \
    --image_path "$IMAGE_DIR" \
    --input_path "$SPARSE_DIR/0" \
    --output_path "$DENSE_DIR" \
    --output_type COLMAP \
    --max_image_size 2000

# Step 5: Dense Stereo (GPU)
colmap patch_match_stereo \
    --workspace_path "$DENSE_DIR" \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

# Step 6: Dense Fusion
colmap stereo_fusion \
    --workspace_path "$DENSE_DIR" \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path "$DENSE_DIR/fused.ply"

