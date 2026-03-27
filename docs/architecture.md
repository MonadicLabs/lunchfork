# lunchfork — Architecture Overview

## System Purpose

lunchfork is a passive RF localisation system for non-cooperative VHF/UHF emitters. It combines:

- **Distributed RSSI collection** from heterogeneous SDR nodes (UAV, ground, vehicles)
- **Conditional diffusion inference** for radio map reconstruction
- **Multi-target particle filter** for tracking
- **Real-time Leaflet.js WebUI**

## Precision targets

| Band | Nodes | CEP50 |
|------|-------|-------|
| UHF (400–900 MHz) | 2–3 UAV orbits | 20–50 m |
| VHF (100–300 MHz) | 2–3 UAV orbits | 80–200 m |

---

## Service architecture

```
                         MQTT broker (Mosquitto)
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    node-sitl-uav1      node-sitl-ground1    node-hitl
    (orbit trajectory)  (static)             (SoapySDR)
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │  rssi/{node_id}
                              ▼
                           master
                    ┌─────────────────┐
                    │ FreqClusterer   │
                    │ SlidingWindow   │
                    │ GeoPreprocessor │
                    │ DiffusionModel  │
                    │ ParticleFilter  │
                    │ WebUI (Leaflet) │
                    └────────┬────────┘
                             │
                    sim-engine (RF propagation)
```

## Message flow

1. **node-sitl** queries **sim-engine** `/rssi` at its current trajectory position
2. **node-sitl** publishes `RssiMessage` to `rssi/{node_id}` on MQTT
3. **master** receives, clusters by frequency, adds to sliding window
4. **master** (Phase 3+) runs diffusion inference + particle filter update
5. **master** pushes `TargetLocation` and `RadioMapUpdate` to WebSocket clients
6. **WebUI** displays nodes, targets, radio map heatmap

## Component responsibilities

### master (x86 only)
- All inference (diffusion, SR, particle filter)
- REST API + WebSocket + WebUI
- Frequency clustering, sliding window management

### node-hitl (ARM)
- SoapySDR → FFT → peak detection → RSSI
- GPS timestamping
- CommTransport publish only
- **Zero ML dependencies**

### sim-engine
- Friis free-space propagation
- Simplified ITM/Longley-Rice for terrain-aware propagation
- Radio map generation for dataset creation

### shared library
- `shared/messages`: Pydantic v2 message schemas
- `shared/comm`: CommTransport abstraction (MQTT, WebSocket)
- `shared/geo`: BBox, GeoPreprocessor, SRTM fetch
- `shared/models`: InferenceBackend, DiffusionModel, SuperResolutionModel

## Deployment environments

| Env | Runtime | Notes |
|-----|---------|-------|
| dev | Docker Compose | Full stack with hot-reload |
| prod-master | Bare metal x86, systemd | onnxruntime-cpu or CUDA |
| prod-node | Bare metal ARM, systemd | No ML, ~150MB RAM |

## Inference pipeline (master, Phase 3+)

```
snapshot = sliding_window.get_snapshot(channel_id)

# Coarse pass
cond_c = geo.get_conditioning_tensor(zone_large, GRID_COARSE_SIZE)
rm_c   = diffusion.infer(cond_c, to_sparse(snapshot, zone_large))
pf.update(channel_id, rm_c, zone_large)

# Fine pass (when PF has converged)
if pf.position_std_m(channel_id) < ZOOM_TRIGGER_STD_M:
    bbox_fine = pf.get_confidence_bbox(channel_id, sigma=2.0)
    cond_f  = geo.get_conditioning_tensor(bbox_fine, GRID_FINE_SIZE)
    rm_f    = diffusion.infer(cond_f, to_sparse(snapshot, bbox_fine))
    mnt_hr  = geo.get_mnt_hires(bbox_fine, GRID_FINE_SIZE * SR_FACTOR)
    rm_hr   = sr_model.upscale(rm_f, mnt_hr)
    pf.update(channel_id, rm_hr, bbox_fine)
```
