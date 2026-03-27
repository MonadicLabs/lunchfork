# ARM-compatible Dockerfile for node-hitl
# No ML dependencies (no pytorch, no onnxruntime)
# SoapySDR installed as system package

FROM python:3.11-slim

WORKDIR /app

# System dependencies (SoapySDR must be installed on the host/system for HITL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libusb-1.0-0 \
    gpsd \
    gpsd-clients \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (no ML)
COPY containers/node-hitl/requirements.txt /app/requirements-hitl.txt
RUN pip install --no-cache-dir -r /app/requirements-hitl.txt

# Install shared library (comm + messages only — no geo/models needed)
COPY shared/ /app/shared/
RUN pip install --no-cache-dir -e /app/shared/

# Copy application
COPY containers/node-hitl/main.py /app/main.py

CMD ["python", "main.py"]
