FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt \
    && rm requirements.txt