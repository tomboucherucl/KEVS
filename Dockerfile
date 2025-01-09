FROM nvidia/cuda:12.0.0-devel-ubuntu20.04
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    software-properties-common sudo && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-distutils curl tzdata && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

RUN apt-get install -y libgl1-mesa-glx libglib2.0-0
ARG USERNAME
ARG UID
ARG GID
RUN groupadd -g $GID $USERNAME && \
    useradd -m -u $UID -g $GID -s /bin/bash $USERNAME
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN chown -R $USERNAME:$USERNAME /app
USER $USERNAME
COPY --chown=$USERNAME:$USERNAME requirements.txt .
ENV BASE="/app/U-Mamba/data" \
    nnUNet_raw="/app/U-Mamba/data/nnUNet_raw" \
    nnUNet_preprocessed="/app/U-Mamba/data/nnUNet_preprocessed" \
    nnUNet_results="/app/U-Mamba/data/nnUNet_results" \
    PATH="/home/$USERNAME/.local/bin:$PATH"

RUN python3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN python3.11 -m pip install packaging
RUN python3.11 -m pip install -r requirements.txt

# Command to run the application
CMD ["bash"]
