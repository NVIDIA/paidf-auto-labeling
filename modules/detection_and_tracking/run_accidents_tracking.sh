#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Script to run RFDETR tracking on all videos in input directory

INPUT_DIR="./input"
OUTPUT_DIR="./output"

echo "=============================================="
echo "Running RFDETR + Deep-OC-SORT / ByteTrack Tracking"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="

# Run the tracking script with ByteTrack
python ./rfdetr_tracking.py \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    --tracker deepocsort \
    --threshold 0.2 \
    --iou-threshold 0.3 \
    --classes car truck bus motorcycle bicycle person\
    --per-class \
    --asso-func diou \
    --min-hits 3 \
    --max-age 60 \
    --min-track-frames 5 \
    --deepocsort-stage2-off \
    --deepocsort-min-hits-nonconsecutive \
    --save-vis \
    --save-video \
    --save-video-red-id \
    --cross-class-iou-threshold 0.9 \
    --dedup-iou-threshold 0.3 \
    --dedup-priority prev_iou \

echo "=============================================="
echo "Processing complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
