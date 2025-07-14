# Use the official Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the application code into the container
COPY . .

# Make the startup script executable
RUN chmod +x runpod-startup.sh

# Set environment variables that will be passed from RunPod
# An empty default value is provided, but these should be set in the RunPod template.
ENV LLM_MODEL_NAME="qwen-plus"
ENV LLM_API_KEY=""
ENV LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
ENV ELEVENLABS_KEY=""
ENV HF_TOKEN=""

# Set the command to run the startup script
CMD ["./runpod-startup.sh"]