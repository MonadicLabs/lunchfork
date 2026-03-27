#!/usr/bin/env bash
# deploy/install-master.sh — Install lunchfork master on bare metal x86
#
# Installs: python3.11, onnxruntime (CPU or CUDA), systemd services, Mosquitto
#
# Usage:
#   sudo bash deploy/install-master.sh
#   sudo bash deploy/install-master.sh --with-gpu   (installs onnxruntime-gpu)
#
# Config: edit /etc/lunchfork/config.env after installation

set -euo pipefail

INSTALL_DIR="/opt/lunchfork"
CONFIG_DIR="/etc/lunchfork"
LOG_DIR="/var/log/lunchfork"
SYSTEMD_DIR="/etc/systemd/system"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

WITH_GPU=false
if [[ "${1:-}" == "--with-gpu" ]]; then
    WITH_GPU=true
fi

echo "=== lunchfork master installation ==="
echo "Root: ${ROOT_DIR}"
echo "Install dir: ${INSTALL_DIR}"
echo "GPU support: ${WITH_GPU}"
echo ""

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    mosquitto \
    mosquitto-clients \
    git \
    curl \
    wget

# ---------------------------------------------------------------------------
# Python virtual environment
# ---------------------------------------------------------------------------
echo "[2/6] Creating Python virtual environment..."
python3.11 -m venv "${INSTALL_DIR}/venv"
source "${INSTALL_DIR}/venv/bin/activate"
pip install --upgrade pip wheel

# ---------------------------------------------------------------------------
# ONNX Runtime
# ---------------------------------------------------------------------------
echo "[3/6] Installing ONNX Runtime..."
if [[ "${WITH_GPU}" == "true" ]]; then
    # Check CUDA availability
    if command -v nvidia-smi &> /dev/null; then
        echo "  GPU detected — installing onnxruntime-gpu"
        pip install onnxruntime-gpu
    else
        echo "  WARN: --with-gpu specified but no GPU detected, falling back to CPU"
        pip install onnxruntime
    fi
else
    pip install onnxruntime
fi

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
echo "[4/6] Installing Python dependencies..."
pip install \
    fastapi \
    uvicorn[standard] \
    pydantic>=2.0 \
    structlog \
    numpy \
    paho-mqtt \
    websockets \
    aiofiles \
    aiohttp \
    pyyaml

# Install shared library
pip install -e "${ROOT_DIR}/shared/"

# Install master container deps
pip install -r "${ROOT_DIR}/containers/master/requirements.txt"

# ---------------------------------------------------------------------------
# Application files
# ---------------------------------------------------------------------------
echo "[5/6] Installing application files..."
mkdir -p "${INSTALL_DIR}/app"
cp -r "${ROOT_DIR}/containers/master/"* "${INSTALL_DIR}/app/"
cp -r "${ROOT_DIR}/shared" "${INSTALL_DIR}/app/shared"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
mkdir -p "${CONFIG_DIR}"
if [[ ! -f "${CONFIG_DIR}/config.env" ]]; then
    cp "${SCRIPT_DIR}/config.master.env" "${CONFIG_DIR}/config.env"
    echo "  Config template copied to ${CONFIG_DIR}/config.env"
    echo "  IMPORTANT: Edit ${CONFIG_DIR}/config.env before starting services"
fi

# Mosquitto configuration
if [[ ! -f "/etc/mosquitto/conf.d/lunchfork.conf" ]]; then
    mkdir -p /etc/mosquitto/conf.d/
    cp "${SCRIPT_DIR}/mosquitto.conf" /etc/mosquitto/conf.d/lunchfork.conf
fi

# ---------------------------------------------------------------------------
# Systemd services
# ---------------------------------------------------------------------------
echo "[6/6] Installing systemd services..."
mkdir -p "${LOG_DIR}"

# Adjust paths in service files and install
for service_file in "${SCRIPT_DIR}/systemd/"*.service; do
    svc_name=$(basename "${service_file}")
    sed "s|/opt/lunchfork|${INSTALL_DIR}|g; s|/etc/lunchfork|${CONFIG_DIR}|g" \
        "${service_file}" > "${SYSTEMD_DIR}/${svc_name}"
    echo "  Installed ${SYSTEMD_DIR}/${svc_name}"
done

systemctl daemon-reload
systemctl enable mosquitto
systemctl enable lunchfork-master

# ---------------------------------------------------------------------------
# Detect GPU and update config
# ---------------------------------------------------------------------------
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "  Set INFERENCE_BACKEND=onnx-cuda in ${CONFIG_DIR}/config.env"
fi

echo ""
echo "=== Installation complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit ${CONFIG_DIR}/config.env (COMM_BROKER_URL, INFERENCE_BACKEND, etc.)"
echo "  2. Fetch terrain: python3 ${ROOT_DIR}/scripts/fetch_terrain.py --source srtm --bbox 4.0 43.0 6.5 44.5"
echo "  3. Start services: systemctl start mosquitto lunchfork-master"
echo "  4. Check logs: journalctl -u lunchfork-master -f"
echo "  5. WebUI: http://localhost:8080"
