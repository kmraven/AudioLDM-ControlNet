#!/bin/bash

# Script to test MotionBERT feature extraction
# Run from AudioLDM-ControlNet directory

echo "Testing MotionBERT Feature Extraction"
echo "======================================"
echo ""

# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Activate conda environment and run test
conda run -n audioldm_train python test_motionbert_simple.py
