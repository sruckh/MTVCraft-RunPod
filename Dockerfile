# Use an official NVIDIA CUDA 12.1.1 development image as the base
# The -devel tag is crucial as it includes the full CUDA Toolkit needed for compilation.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.10, pip, and other basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the application code into the container
COPY . .

# Make the startup script executable
RUN chmod +x runpod-startup.sh

# Set environment variables that will be passed from RunPod
ENV LLM_MODEL_NAME="qwen-plus"
ENV LLM_API_KEY=""
ENV LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
ENV ELEVENLABS_KEY=""
ENV HF_TOKEN=""

# Set the command to run the startup script
# This script will install dependencies and launch the app
CMD ["./runpod-startup.sh"]

