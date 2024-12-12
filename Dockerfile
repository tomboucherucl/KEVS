# Use an official NVIDIA CUDA runtime image as a base
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set non-interactive frontend for tzdata
ENV DEBIAN_FRONTEND=noninteractive

# Define build arguments (with defaults)
ARG USERNAME=defaultuser
ARG UID=1000
ARG GID=1000
ARG BASE_PATH=/app/data  # Default path inside the container

# Install Python 3.11 and other dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common sudo && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-distutils curl tzdata && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

# Install OpenCV dependencies
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0

# Create user (for permissions on mounted volumes)
RUN groupadd -g $GID $USERNAME && \
    useradd -m -u $UID -g $GID -s /bin/bash $USERNAME

# Grant sudo (if absolutely necessary)
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set working directory
WORKDIR /app

# Set environment variables for paths (using ARG)
ENV BASE="${BASE_PATH}" \
    nnUNet_raw="${BASE_PATH}/nnUNet_raw" \
    nnUNet_preprocessed="${BASE_PATH}/nnUNet_preprocessed" \
    nnUNet_results="${BASE_PATH}/nnUNet_results"

# Copy the application code (make sure user owns files)
COPY --chown=$USERNAME:$USERNAME . /app

# Install PyTorch
RUN python3.11 -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

RUN python3.11 -m pip install packaging

# Install Python dependencies
RUN python3.11 -m pip install -r requirements.txt

# Switch to non-root user
USER $USERNAME

# Command to run (default: bash for interactive use)
CMD ["bash"]