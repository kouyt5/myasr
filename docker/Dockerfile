ARG BASE_IMAGE=pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel
FROM $BASE_IMAGE
# GPA报错bug
# RUN echo "Acquire::http::Proxy \"http://chencx.cn:8888\";" > /etc/apt/apt.conf
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt \
    && rm requirements.txt \
    && apt-get update && apt install sox -y
# apex
RUN apt install git -y \
    && git clone https://github.com/NVIDIA/apex \
    && cd apex \
    && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ \
    && rm -r ../apex

# install utilities and dependencies
RUN pip install awscli --upgrade ruamel.yaml visdom h5py
WORKDIR /workspace
USER root
ENTRYPOINT ["python", "-m", "torchelastic.distributed.launch"]
CMD ["--help"]
