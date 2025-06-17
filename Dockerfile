FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY builder/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy handler
COPY src/handler.py .

# Required by RunPod Serverless
CMD ["python3", "-u", "handler.py"]
