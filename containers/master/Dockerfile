FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install master requirements
COPY containers/master/requirements.txt /app/requirements-master.txt
RUN pip install --no-cache-dir -r /app/requirements-master.txt

# Install shared library
COPY shared/ /app/shared/
RUN pip install --no-cache-dir -e /app/shared/

# Copy application
COPY containers/master/main.py /app/main.py
COPY containers/master/pipeline/ /app/pipeline/
COPY containers/master/static/ /app/static/

EXPOSE 8080

CMD ["python", "main.py"]
