FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt \
    && rm requirements.txt \
    && apt update && apt install sox -y
# apex
RUN apt install git -y \
    && git clone https://github.com/NVIDIA/apex \
    && cd apex \
    && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ \
    && rm -r ../apex