#!/usr/bin/env bash
# deploy/install-node.sh — Install lunchfork node-hitl on ARM bare metal
#
# Compatible with: RPi 4/5, CM4, Jetson Nano (ARM64)
# NO pytorch, NO onnxruntime — minimal footprint (~150MB RAM)
#
# Installs: python3.11, soapysdr, gpsd, paho-mqtt, lunchfork-node-hitl
#
# Usage:
#   sudo bash deploy/install-node.sh
#
# Config: edit /etc/lunchfork/config.env after installation

set -euo pipefail

INSTALL_DIR="/opt/lunchfork"
CONFIG_DIR="/etc/lunchfork"
LOG_DIR="/var/log/lunchfork"
SYSTEMD_DIR="/etc/systemd/system"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== lunchfork node-hitl installation (ARM) ==="
echo "Root: ${ROOT_DIR}"
echo "Install dir: ${INSTALL_DIR}"
echo ""

# Verify we're not accidentally installing ML deps
echo "[0/6] Pre-flight checks..."
if python3 -c "import torch" 2>/dev/null; then
    echo "  WARN: PyTorch is installed on this system."
    echo "  node-hitl does not require it — this is unusual for an ARM node."
fi

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
    libusb-1.0-0-dev \
    gpsd \
    gpsd-clients \
    git \
    # SoapySDR and RTL-SDR
    soapysdr-tools \
    libsoapysdr-dev \
    soapysdr-module-rtlsdr \
    rtl-sdr \
    # Disable RTL-SDR kernel module (conflicts with SoapySDR)
    || true

# Blacklist DVB kernel module that conflicts with RTL-SDR
if [[ ! -f /etc/modprobe.d/blacklist-rtl.conf ]]; then
    echo "blacklist dvb_usb_rtl28xxu" > /etc/modprobe.d/blacklist-rtl.conf
    echo "  Blacklisted dvb_usb_rtl28xxu (needed for RTL-SDR)"
fi

# Optimise CPU frequency for RPi (optional)
if [[ -f /boot/config.txt ]]; then
    if ! grep -q "force_turbo=1" /boot/config.txt; then
        echo ""
        echo "  TIP: For best FFT performance on RPi, add to /boot/config.txt:"
        echo "    force_turbo=1"
        echo "    arm_freq=2000  (RPi 4)"
    fi
fi

# ---------------------------------------------------------------------------
# Python virtual environment (NO ML deps)
# ---------------------------------------------------------------------------
echo "[2/6] Creating Python virtual environment (no ML)..."
python3.11 -m venv "${INSTALL_DIR}/venv"
source "${INSTALL_DIR}/venv/bin/activate"
pip install --upgrade pip wheel

# CRITICAL: do NOT install pytorch or onnxruntime
echo "[3/6] Installing Python dependencies (node-only, no ML)..."
pip install \
    numpy \
    structlog \
    pydantic>=2.0 \
    paho-mqtt \
    gpsd-py3 \
    pyserial \
    aiohttp

# Shared library (comm + messages only)
pip install -e "${ROOT_DIR}/shared/"

# ---------------------------------------------------------------------------
# Application files
# ---------------------------------------------------------------------------
echo "[4/6] Installing application files..."
mkdir -p "${INSTALL_DIR}/app"
cp "${ROOT_DIR}/containers/node-hitl/main.py" "${INSTALL_DIR}/app/"
cp -r "${ROOT_DIR}/shared" "${INSTALL_DIR}/app/shared"

# ---------------------------------------------------------------------------
# GPS configuration
# ---------------------------------------------------------------------------
echo "[5/6] Configuring gpsd..."
cat > /etc/default/gpsd << 'EOF'
START_DAEMON="true"
GPSD_OPTIONS="-n"
DEVICES="/dev/ttyACM0"
USBAUTO="true"
SOCKET="/var/run/gpsd.sock"
EOF

systemctl enable gpsd
echo "  gpsd configured for /dev/ttyACM0 (adjust DEVICES for your GPS)"

# ---------------------------------------------------------------------------
# Configuration & systemd
# ---------------------------------------------------------------------------
echo "[6/6] Installing configuration and systemd services..."
mkdir -p "${CONFIG_DIR}" "${LOG_DIR}"

if [[ ! -f "${CONFIG_DIR}/config.env" ]]; then
    cp "${SCRIPT_DIR}/config.node.env" "${CONFIG_DIR}/config.env"
    echo "  Config template copied to ${CONFIG_DIR}/config.env"
    echo "  IMPORTANT: Edit NODE_ID, COMM_BROKER_URL, GPS_SOURCE in config.env"
fi

# Install systemd service
svc_file="${SCRIPT_DIR}/systemd/lunchfork-node-hitl.service"
if [[ -f "${svc_file}" ]]; then
    sed "s|/opt/lunchfork|${INSTALL_DIR}|g; s|/etc/lunchfork|${CONFIG_DIR}|g" \
        "${svc_file}" > "${SYSTEMD_DIR}/lunchfork-node-hitl.service"
    systemctl daemon-reload
    systemctl enable lunchfork-node-hitl
    echo "  Service installed: lunchfork-node-hitl"
fi

echo ""
echo "=== Installation complete ==="
echo ""
echo "Memory footprint estimate: ~150MB RAM (node-hitl idle)"
echo ""
echo "Next steps:"
echo "  1. Edit ${CONFIG_DIR}/config.env:"
echo "     - Set NODE_ID to a unique identifier (e.g. node-$(hostname))"
echo "     - Set COMM_BROKER_URL to point to your master (mqtt://MASTER_IP:1883)"
echo "     - Set SDR_DRIVER (rtlsdr/hackrf/limesdr)"
echo "     - Set GPS_SOURCE (gpsd/nmea/static)"
echo "  2. Connect RTL-SDR and GPS receiver"
echo "  3. Start: systemctl start lunchfork-node-hitl"
echo "  4. Check: journalctl -u lunchfork-node-hitl -f"
