FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install node-sitl requirements
COPY containers/node-sitl/requirements.txt /app/requirements-node.txt
RUN pip install --no-cache-dir -r /app/requirements-node.txt

# Install shared library
COPY shared/ /app/shared/
RUN pip install --no-cache-dir -e /app/shared/

# Copy application
COPY containers/node-sitl/main.py /app/main.py

CMD ["python", "main.py"]
