FROM nvidia/cuda:12.6.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 libglib2.0-dev \
    libgtk-3-0 libgtk-3-dev \
    libcudnn8 \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

RUN pip3 install numpy opencv-python

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /unet

CMD ["/bin/bash"]