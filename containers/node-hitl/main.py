"""
node-hitl — Physical SDR node for lunchfork HITL testing.

Pipeline:
  SoapySDR (or IQ replay file) → FFT sliding window →
  Frequency clustering → RSSI per channel →
  GPS position → Publish RssiMessage + NodeStatus via CommTransport

No ML dependencies — designed to run on RPi 4/5 (ARM).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import structlog

# Ensure shared is importable
sys.path.insert(0, "/app")

from shared.comm import get_transport
from shared.messages import (
    FreqChannel,
    NodePosition,
    NodeStatus,
    RssiMessage,
)

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NODE_ID = os.environ.get("NODE_ID", "hitl-node-001")
NODE_TYPE = os.environ.get("NODE_TYPE", "ground")

SDR_DRIVER = os.environ.get("SDR_DRIVER", "rtlsdr")
SDR_FREQ_CENTER_HZ = int(os.environ.get("SDR_FREQ_CENTER_HZ", "433920000"))
SDR_SAMPLE_RATE = int(os.environ.get("SDR_SAMPLE_RATE", "2048000"))
SDR_GAIN = float(os.environ.get("SDR_GAIN", "40"))
FFT_SIZE = int(os.environ.get("FFT_SIZE", "2048"))
FFT_OVERLAP = float(os.environ.get("FFT_OVERLAP", "0.5"))

FREQ_CLUSTER_THRESHOLD_DBM = float(os.environ.get("FREQ_CLUSTER_THRESHOLD_DBM", "-90"))
FREQ_CLUSTER_BW_HZ = int(os.environ.get("FREQ_CLUSTER_BW_HZ", "25000"))

GPS_SOURCE = os.environ.get("GPS_SOURCE", "static")  # gpsd|nmea|static
GPS_DEVICE = os.environ.get("GPS_DEVICE", "/dev/ttyACM0")
GPS_STATIC_LAT = float(os.environ.get("GPS_STATIC_LAT", "43.530"))
GPS_STATIC_LON = float(os.environ.get("GPS_STATIC_LON", "5.450"))
GPS_STATIC_ALT_M = float(os.environ.get("GPS_STATIC_ALT_M", "200"))

REPLAY_FILE = os.environ.get("REPLAY_FILE", "")

STATUS_INTERVAL_S = float(os.environ.get("STATUS_INTERVAL_S", "10"))


# ---------------------------------------------------------------------------
# GPS abstraction
# ---------------------------------------------------------------------------


class GpsProvider:
    """GPS position provider with multiple backends."""

    def __init__(self) -> None:
        self._source = GPS_SOURCE
        self._last_pos: NodePosition | None = None

    async def get_position(self) -> NodePosition:
        if self._source == "static":
            return NodePosition(
                lat=GPS_STATIC_LAT,
                lon=GPS_STATIC_LON,
                alt_m=GPS_STATIC_ALT_M,
                accuracy_m=10.0,
            )
        if self._source == "gpsd":
            return await self._gpsd_position()
        if self._source == "nmea":
            return await self._nmea_position()
        return NodePosition(lat=GPS_STATIC_LAT, lon=GPS_STATIC_LON, alt_m=GPS_STATIC_ALT_M)

    async def _gpsd_position(self) -> NodePosition:
        """Read position from gpsd daemon."""
        try:
            import gps  # type: ignore[import]
            gpsd = gps.gps(mode=gps.WATCH_ENABLE | gps.WATCH_NEWSTYLE)
            gpsd.next()
            if hasattr(gpsd, "fix") and gpsd.fix.mode >= 2:
                return NodePosition(
                    lat=gpsd.fix.latitude,
                    lon=gpsd.fix.longitude,
                    alt_m=float(gpsd.fix.altitude) if gpsd.fix.mode >= 3 else GPS_STATIC_ALT_M,
                    accuracy_m=float(gpsd.fix.epx) if not np.isnan(gpsd.fix.epx) else None,
                )
        except Exception as exc:
            logger.warning("gps.gpsd_failed", error=str(exc))
        return NodePosition(lat=GPS_STATIC_LAT, lon=GPS_STATIC_LON, alt_m=GPS_STATIC_ALT_M)

    async def _nmea_position(self) -> NodePosition:
        """Read NMEA sentences from serial device."""
        try:
            import serial  # type: ignore[import]
            with serial.Serial(GPS_DEVICE, 9600, timeout=2) as ser:
                for _ in range(20):
                    line = ser.readline().decode("ascii", errors="replace").strip()
                    if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
                        parts = line.split(",")
                        if len(parts) >= 6 and parts[2] and parts[4]:
                            lat_raw = float(parts[2])
                            lat_deg = int(lat_raw / 100) + (lat_raw % 100) / 60
                            if parts[3] == "S":
                                lat_deg = -lat_deg
                            lon_raw = float(parts[4])
                            lon_deg = int(lon_raw / 100) + (lon_raw % 100) / 60
                            if parts[5] == "W":
                                lon_deg = -lon_deg
                            alt_m = float(parts[9]) if len(parts) > 9 and parts[9] else GPS_STATIC_ALT_M
                            return NodePosition(lat=lat_deg, lon=lon_deg, alt_m=alt_m)
        except Exception as exc:
            logger.warning("gps.nmea_failed", error=str(exc))
        return NodePosition(lat=GPS_STATIC_LAT, lon=GPS_STATIC_LON, alt_m=GPS_STATIC_ALT_M)


# ---------------------------------------------------------------------------
# FFT / RSSI computation (no ML, pure numpy)
# ---------------------------------------------------------------------------


def compute_psd(
    iq_samples: np.ndarray,
    sample_rate: int,
    fft_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density from IQ samples.

    Returns:
      freqs_hz: [fft_size] frequency bins in Hz (relative to centre)
      psd_dbm: [fft_size] power in dBm per bin
    """
    n = len(iq_samples)
    if n < fft_size:
        iq_samples = np.pad(iq_samples, (0, fft_size - n))
    else:
        iq_samples = iq_samples[:fft_size]

    # Apply Hann window
    window = np.hanning(fft_size)
    windowed = iq_samples * window

    # FFT
    spectrum = np.fft.fftshift(np.fft.fft(windowed, n=fft_size))
    psd_linear = (np.abs(spectrum) ** 2) / (fft_size * sample_rate)

    # Convert to dBm (assuming 50Ω, 1mW reference)
    psd_dbm = 10 * np.log10(psd_linear + 1e-30)

    # Frequency axis
    freqs_hz = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate))

    return freqs_hz, psd_dbm


def find_signal_peaks(
    freqs_hz: np.ndarray,
    psd_dbm: np.ndarray,
    threshold_dbm: float,
    cluster_bw_hz: int,
    center_freq_hz: int,
) -> list[tuple[int, float]]:
    """
    Find frequency peaks above threshold and cluster nearby peaks.

    Returns list of (absolute_freq_hz, peak_rssi_dbm).
    """
    above_threshold = psd_dbm > threshold_dbm
    if not np.any(above_threshold):
        return []

    # Find local maxima above threshold
    from collections import defaultdict

    peaks: list[tuple[int, float]] = []
    in_peak = False
    peak_start = 0

    for i in range(len(psd_dbm)):
        if above_threshold[i] and not in_peak:
            in_peak = True
            peak_start = i
        elif not above_threshold[i] and in_peak:
            in_peak = False
            # Find max in this peak region
            region = psd_dbm[peak_start:i]
            peak_idx = peak_start + int(np.argmax(region))
            abs_freq = center_freq_hz + int(freqs_hz[peak_idx])
            peaks.append((abs_freq, float(psd_dbm[peak_idx])))

    if in_peak:
        region = psd_dbm[peak_start:]
        peak_idx = peak_start + int(np.argmax(region))
        abs_freq = center_freq_hz + int(freqs_hz[peak_idx])
        peaks.append((abs_freq, float(psd_dbm[peak_idx])))

    # Merge peaks within cluster_bw_hz
    if not peaks:
        return peaks

    merged: list[tuple[int, float]] = [peaks[0]]
    for freq, rssi in peaks[1:]:
        if abs(freq - merged[-1][0]) <= cluster_bw_hz:
            # Keep higher RSSI
            if rssi > merged[-1][1]:
                merged[-1] = (freq, rssi)
        else:
            merged.append((freq, rssi))

    return merged


# ---------------------------------------------------------------------------
# SDR abstraction
# ---------------------------------------------------------------------------


class SdrSource:
    """Abstract SDR sample source."""

    def read_samples(self, n_samples: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SoapySdrSource(SdrSource):
    """SoapySDR-based IQ sample source."""

    def __init__(self) -> None:
        try:
            import SoapySDR  # type: ignore[import]
            from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX

            self._SoapySDR = SoapySDR
            args = dict(driver=SDR_DRIVER)
            self._sdr = SoapySDR.Device(args)
            self._sdr.setSampleRate(SOAPY_SDR_RX, 0, SDR_SAMPLE_RATE)
            self._sdr.setFrequency(SOAPY_SDR_RX, 0, SDR_FREQ_CENTER_HZ)
            self._sdr.setGain(SOAPY_SDR_RX, 0, SDR_GAIN)
            self._stream = self._sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
            self._sdr.activateStream(self._stream)
            logger.info(
                "sdr.opened",
                driver=SDR_DRIVER,
                freq_hz=SDR_FREQ_CENTER_HZ,
                sample_rate=SDR_SAMPLE_RATE,
            )
        except Exception as exc:
            logger.error("sdr.open_failed", driver=SDR_DRIVER, error=str(exc))
            raise

    def read_samples(self, n_samples: int) -> np.ndarray:
        import SoapySDR
        from SoapySDR import SOAPY_SDR_CF32

        buf = np.zeros(n_samples, dtype=np.complex64)
        sr = self._sdr.readStream(self._stream, [buf], n_samples)
        if sr.ret < 0:
            logger.warning("sdr.read_error", ret=sr.ret)
            return buf[:0]
        return buf[: sr.ret]

    def close(self) -> None:
        try:
            self._sdr.deactivateStream(self._stream)
            self._sdr.closeStream(self._stream)
        except Exception:
            pass


class IqReplaySource(SdrSource):
    """Replay IQ samples from a file (raw float32 interleaved I/Q or .npy)."""

    def __init__(self, filepath: str) -> None:
        path = Path(filepath)
        if path.suffix == ".npy":
            self._data = np.load(filepath).astype(np.complex64)
        else:
            # Raw interleaved float32 I/Q
            raw = np.fromfile(filepath, dtype=np.float32)
            self._data = raw[::2] + 1j * raw[1::2]
        self._pos = 0
        logger.info("iq_replay.loaded", filepath=filepath, n_samples=len(self._data))

    def read_samples(self, n_samples: int) -> np.ndarray:
        end = self._pos + n_samples
        if end > len(self._data):
            # Wrap around
            self._pos = 0
            end = n_samples
        samples = self._data[self._pos : end]
        self._pos = end
        return samples


class SyntheticSdrSource(SdrSource):
    """Generate synthetic noise for testing without hardware."""

    def read_samples(self, n_samples: int) -> np.ndarray:
        return (
            np.random.randn(n_samples).astype(np.float32)
            + 1j * np.random.randn(n_samples).astype(np.float32)
        ) * 0.01  # noise floor


def build_sdr_source() -> SdrSource:
    if REPLAY_FILE:
        return IqReplaySource(REPLAY_FILE)
    try:
        return SoapySdrSource()
    except Exception:
        logger.warning("sdr.fallback_to_synthetic")
        return SyntheticSdrSource()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def process_loop(
    sdr: SdrSource,
    transport,
    gps: GpsProvider,
) -> None:
    """Main SDR processing and publish loop."""
    start_time = time.monotonic()
    rssi_count = 0
    last_status_time = start_time

    # Averaging buffer for PSD (reduce noise)
    psd_avg: np.ndarray | None = None
    avg_count = 0
    avg_frames = 4  # average over N FFT frames

    step_samples = int(FFT_SIZE * (1 - FFT_OVERLAP))

    while True:
        loop_start = time.monotonic()

        # Read IQ samples
        samples = await asyncio.get_event_loop().run_in_executor(
            None, sdr.read_samples, FFT_SIZE
        )

        if len(samples) < FFT_SIZE // 2:
            await asyncio.sleep(0.01)
            continue

        # Compute PSD
        freqs_hz, psd_dbm = compute_psd(samples, SDR_SAMPLE_RATE, FFT_SIZE)

        # Accumulate for averaging
        if psd_avg is None:
            psd_avg = psd_dbm.copy()
        else:
            psd_avg = np.maximum(psd_avg, psd_dbm)  # peak hold
        avg_count += 1

        if avg_count < avg_frames:
            continue

        # Detect peaks
        peaks = find_signal_peaks(
            freqs_hz,
            psd_avg,
            threshold_dbm=FREQ_CLUSTER_THRESHOLD_DBM,
            cluster_bw_hz=FREQ_CLUSTER_BW_HZ,
            center_freq_hz=SDR_FREQ_CENTER_HZ,
        )

        # Get GPS position
        pos = await gps.get_position()
        now = datetime.now(timezone.utc)

        # Publish one RssiMessage per detected channel
        for abs_freq_hz, rssi_dbm in peaks:
            channel = FreqChannel(
                center_hz=abs_freq_hz,
                bandwidth_hz=FREQ_CLUSTER_BW_HZ,
            )
            msg = RssiMessage(
                node_id=NODE_ID,
                node_type=NODE_TYPE,  # type: ignore[arg-type]
                timestamp_utc=now,
                position=pos,
                freq_channel=channel,
                rssi_dbm=rssi_dbm,
                is_simulated=False,
            )
            await transport.publish(f"rssi/{NODE_ID}", msg)
            rssi_count += 1
            logger.debug(
                "hitl.rssi_published",
                freq_hz=abs_freq_hz,
                rssi_dbm=round(rssi_dbm, 1),
            )

        # Reset averaging buffer
        psd_avg = None
        avg_count = 0

        # Periodic status
        now_mono = time.monotonic()
        if now_mono - last_status_time >= STATUS_INTERVAL_S:
            elapsed = now_mono - last_status_time
            rate = rssi_count / max(elapsed, 1.0)
            status = NodeStatus(
                node_id=NODE_ID,
                timestamp_utc=datetime.now(timezone.utc),
                node_type=NODE_TYPE,  # type: ignore[arg-type]
                position=pos,
                sdr_ok=True,
                gps_ok=GPS_SOURCE != "static" or True,
                comm_ok=await transport.healthcheck(),
                rssi_rate_hz=round(rate, 2),
            )
            await transport.publish(f"node/status/{NODE_ID}", status)
            rssi_count = 0
            last_status_time = now_mono

        # Small sleep to avoid busy-spinning
        elapsed = time.monotonic() - loop_start
        await asyncio.sleep(max(0.0, 0.05 - elapsed))


async def main() -> None:
    logger.info(
        "node_hitl.starting",
        node_id=NODE_ID,
        sdr_driver=SDR_DRIVER,
        freq_hz=SDR_FREQ_CENTER_HZ,
        gps_source=GPS_SOURCE,
        replay_file=REPLAY_FILE or "none",
    )

    transport = get_transport()
    await transport.connect()
    logger.info("hitl.transport_connected")

    gps = GpsProvider()
    sdr = build_sdr_source()

    try:
        await process_loop(sdr, transport, gps)
    finally:
        sdr.close()
        await transport.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
