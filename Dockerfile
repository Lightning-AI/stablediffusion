from ghcr.io/gridai/base-images:v1.19-gpu-launcher-0.0.43-lightning-1.8.6

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt