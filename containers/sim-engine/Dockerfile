FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install sim-engine requirements
COPY containers/sim-engine/requirements.txt /app/requirements-engine.txt
RUN pip install --no-cache-dir -r /app/requirements-engine.txt

# Install shared library in editable mode
COPY shared/ /app/shared/
RUN pip install --no-cache-dir -e /app/shared/

# Copy application
COPY containers/sim-engine/main.py /app/main.py

EXPOSE 9000

CMD ["python", "main.py"]
