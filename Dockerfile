FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install Python 3.10
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3-pip git && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY builder/requirements.txt .
RUN python3 -m pip install --upgrade pip && pip install -r requirements.txt

# Copy handler
COPY src/handler.py .

CMD ["python3", "-u", "handler.py"]
