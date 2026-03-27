# lunchfork — ARM Node Deployment Guide

## Supported platforms

| Platform | RAM | Notes |
|----------|-----|-------|
| RPi 4 (2GB+) | 2–8 GB | Recommended minimum |
| RPi 5 | 4–8 GB | Best performance |
| CM4 | 1–8 GB | For custom carrier boards |
| Jetson Nano | 4 GB | Overkill — but works |

## Hardware requirements

- **RTL-SDR** (R820T2 recommended): ~$25
- **GPS receiver** (u-blox NEO-M8N or similar): via USB or UART
- **Antenna**: whip or yagi matched to target band
- **Power**: stable 5V/3A USB-C (RPi 4) or official PSU

## Installation

```bash
# On the RPi (as root)
git clone https://github.com/monadic-labs/lunchfork.git /opt/lunchfork/src
cd /opt/lunchfork/src
sudo bash deploy/install-node.sh
```

## Configuration

Edit `/etc/lunchfork/config.env`:

```env
NODE_ID=node-alpha        # unique per node
COMM_BROKER_URL=mqtt://192.168.1.100:1883   # master IP
SDR_DRIVER=rtlsdr
SDR_FREQ_CENTER_HZ=433920000
GPS_SOURCE=gpsd
```

## RTL-SDR setup

```bash
# Verify device detected
rtl_test -t

# Check SoapySDR sees it
SoapySDRUtil --probe
```

If the device is claimed by the DVB kernel module:
```bash
echo "blacklist dvb_usb_rtl28xxu" | sudo tee /etc/modprobe.d/blacklist-rtl.conf
sudo modprobe -r dvb_usb_rtl28xxu
```

## GPS setup

```bash
# Check GPS device path
ls /dev/ttyACM*   # USB GPS
ls /dev/ttyS*     # UART GPS

# Test NMEA sentences
cat /dev/ttyACM0

# Test gpsd
gpsd /dev/ttyACM0 -F /var/run/gpsd.sock
cgps -s
```

## Performance tuning (RPi 4)

Add to `/boot/config.txt`:
```
# Fix CPU at max frequency for stable FFT latency
force_turbo=1
arm_freq=2000
over_voltage=2
```

Reduce GPU memory (not needed for node):
```
gpu_mem=16
```

## Service management

```bash
# Start
sudo systemctl start lunchfork-node-hitl

# Status / logs
sudo systemctl status lunchfork-node-hitl
sudo journalctl -u lunchfork-node-hitl -f

# Stop
sudo systemctl stop lunchfork-node-hitl
```

## Replay mode

To validate the pipeline without hardware, replay a recorded IQ capture:

```bash
# Record IQ samples (rtl-sdr tools)
rtl_sdr -f 433920000 -s 2048000 -n 10000000 capture.iq

# Configure node to replay
echo "REPLAY_FILE=/path/to/capture.iq" >> /etc/lunchfork/config.env
sudo systemctl restart lunchfork-node-hitl
```

## Memory footprint

Expected idle memory usage:
- Python interpreter: ~40 MB
- numpy + paho-mqtt: ~30 MB
- FFT buffers (2048 samples × float32): < 1 MB
- **Total: ~150 MB** — comfortable on RPi 4 2GB
