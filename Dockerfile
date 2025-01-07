# Use an official NVIDIA CUDA runtime image as a base
FROM nvidia/cuda:12.0.0-devel-ubuntu20.04

# Set the working directory inside the container
WORKDIR /app

# Set non-interactive frontend for tzdata
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and other dependencies including tzdata and sudo
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

# Set the timezone environment variable (replace 'America/New_York' with your timezone)
#ENV TZ=Europe/London

# Configure the timezone non-interactively
#RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
#    dpkg-reconfigure --frontend noninteractive tzdata

# Get the host user UID and GID from build arguments
ARG USERNAME
ARG UID
ARG GID

# Create the user inside the container with the same UID and GID
RUN groupadd -g $GID $USERNAME && \
    useradd -m -u $UID -g $GID -s /bin/bash $USERNAME

# Grant sudo privileges to the user without requiring a password
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set the permissions for the working directory
RUN chown -R $USERNAME:$USERNAME /app

# Switch to the non-root user
USER $USERNAME

# Copy the requirements.txt file into the working directory
COPY --chown=$USERNAME:$USERNAME requirements.txt .

# Set environment variables for paths
ENV BASE="/app/U-Mamba/data" \
    nnUNet_raw="/app/U-Mamba/data/nnUNet_raw" \
    nnUNet_preprocessed="/app/U-Mamba/data/nnUNet_preprocessed" \
    nnUNet_results="/app/U-Mamba/data/nnUNet_results" \
    PATH="/home/$USERNAME/.local/bin:$PATH"

# Install PyTorch
RUN python3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN python3.11 -m pip install packaging

# Install the Python dependencies from requirements.txt
RUN python3.11 -m pip install -r requirements.txt

# Command to run the application
CMD ["bash"]
