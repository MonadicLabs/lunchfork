# lunchfork — Message Schemas

All messages are Pydantic v2 models, serialised as JSON over CommTransport.

## Topics

| Topic | Message type | Direction |
|-------|-------------|-----------|
| `rssi/{node_id}` | `RssiMessage` | node → master |
| `node/status/{node_id}` | `NodeStatus` | node → master |
| `loc/target/{target_id}` | `TargetLocation` | master → UI |
| `loc/radiomap/{freq_hz}` | `RadioMapUpdate` | master → UI |
| `sim/emitter/{emitter_id}` | `EmitterState` | sim → log |

## Schemas

### NodePosition
```json
{
  "lat": 43.530,
  "lon": 5.450,
  "alt_m": 120.0,
  "accuracy_m": 5.0
}
```

### FreqChannel
```json
{
  "center_hz": 433920000,
  "bandwidth_hz": 25000,
  "label": "UHF-433"
}
```

### RssiMessage
```json
{
  "version": "1.0",
  "node_id": "sim-uav-1",
  "node_type": "uav",
  "timestamp_utc": "2026-03-24T10:00:00Z",
  "position": { "lat": 43.535, "lon": 5.462, "alt_m": 120.0 },
  "freq_channel": { "center_hz": 433920000, "bandwidth_hz": 25000, "label": "UHF-433" },
  "rssi_dbm": -72.5,
  "snr_db": 12.0,
  "is_simulated": true
}
```

### NodeStatus
```json
{
  "version": "1.0",
  "node_id": "hitl-node-001",
  "timestamp_utc": "2026-03-24T10:00:10Z",
  "node_type": "ground",
  "position": { "lat": 43.510, "lon": 5.430, "alt_m": 180.0 },
  "sdr_ok": true,
  "gps_ok": true,
  "comm_ok": true,
  "rssi_rate_hz": 0.97
}
```

### TargetLocation
```json
{
  "version": "1.0",
  "target_id": "target-1",
  "timestamp_utc": "2026-03-24T10:05:00Z",
  "position": { "lat": 43.530, "lon": 5.450, "alt_m": 5.0 },
  "uncertainty_m": 85.0,
  "covariance": [[7225, 0], [0, 7225]],
  "n_particles": 500,
  "freq_channel": { "center_hz": 433920000, "bandwidth_hz": 25000 },
  "track_state": "confirmed"
}
```

### RadioMapUpdate
```json
{
  "version": "1.0",
  "timestamp_utc": "2026-03-24T10:05:00Z",
  "freq_channel": { "center_hz": 433920000, "bandwidth_hz": 25000 },
  "bbox": { "lat_min": 43.4, "lon_min": 4.5, "lat_max": 43.8, "lon_max": 5.2 },
  "resolution_m_per_px": 30.0,
  "data_b64": "<base64-encoded float32 [H,W] array>"
}
```

### EmitterState
```json
{
  "version": "1.0",
  "id": "target-1",
  "lat": 43.530,
  "lon": 5.450,
  "alt_m": 5.0,
  "freq_hz": 433920000,
  "power_dbm": 10.0,
  "timestamp_utc": "2026-03-24T10:00:00Z"
}
```

## Versioning

- All messages carry a `version` field (currently `"1.0"`)
- Breaking changes require a version bump and a migration period
- Receivers should tolerate unknown fields (Pydantic v2 `model_config = {"extra": "ignore"}`)
