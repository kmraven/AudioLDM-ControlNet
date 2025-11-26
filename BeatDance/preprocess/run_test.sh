#!/bin/bash

# Script to run MotionBERT feature extraction test
# Make sure to activate the conda environment first

echo "=================================="
echo "MotionBERT Feature Extraction Test"
echo "=================================="
echo ""
echo "Usage:"
echo "  conda activate audioldm_train"
echo "  cd /home/sangheon/Desktop/AudioLDM-ControlNet/BeatDance/preprocess"
echo "  bash run_test.sh"
echo ""
echo "Or run directly with conda:"
echo "  conda run -n audioldm_train python test_motionbert_extraction.py"
echo ""

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠ WARNING: No conda environment is activated"
    echo "Attempting to run with conda..."
    echo ""
    conda run -n audioldm_train python test_motionbert_extraction.py
else
    echo "✓ Running in conda environment: $CONDA_DEFAULT_ENV"
    echo ""
    python test_motionbert_extraction.py
fi
