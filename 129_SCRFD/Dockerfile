ARG PYTORCH="1.7.0"
ARG CUDA="11.0"
ARG CUDNN="8"
ARG MM="2.8.0"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update \
&& apt-get install -y \
ffmpeg libsm6 libxext6 git ninja-build \
libglib2.0-0 libsm6 libxrender-dev libxext6 \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
RUN pip install mmdet

RUN conda clean --all
ENV FORCE_CUDA="1"

RUN git clone https://github.com/open-mmlab/mmtracking.git /mmtracking
WORKDIR /mmtracking
RUN pip install -r requirements/build.txt
RUN pip install -v -e .