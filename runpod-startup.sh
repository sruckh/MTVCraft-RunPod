#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# 1. Install system dependencies
echo ">>> Installing system dependencies..."
apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git

# 2. Install Python dependencies
echo ">>> Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt
pip install huggingface_hub[cli]

# 3. Download pre-trained models from Hugging Face
# This requires the HF_TOKEN environment variable to be set in RunPod.
echo ">>> Downloading pre-trained models..."
huggingface-cli download BAAI/MTVCraft --local-dir ./pretrained_models --token $HF_TOKEN

echo ">>> Model download complete. Final directory structure:"
ls -R ./pretrained_models

# 4. Launch the Gradio application
echo ">>> Starting Gradio application..."
bash scripts/app.sh /workspace/OUTPUT

