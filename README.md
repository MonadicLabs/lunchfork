# lunchfork

Système de localisation passive de mobiles RF (VHF/UHF, non-coopératifs) basé sur collecte RSSI distribuée et reconstruction de radio map par réseau de neurones.

**Projet personnel — Monadic Labs.**

---

## Architecture

```
Nœuds SDR (ARM)          Master x86                WebUI
─────────────────         ───────────────────────   ──────────
RTL-SDR / HackRF   →      FreqClusterer             Leaflet.js
FFT + clustering   MQTT   SlidingWindow             Heatmap overlay
GPS horodatage     ────→  UNet / GridLikelihood  →  Cibles + ellipses
                          ParticleFilter            Nœuds actifs
                          WebSocket push
```

- **Transport** : MQTT (défaut), WebSocket (dev), Zenoh (prévu — P2P sans broker)
- **Inférence** : ONNX Runtime CPU (prod), ONNX CUDA / PyTorch CUDA (avec GPU)
- **Nœuds ARM** : zéro dépendance ML — SoapySDR + numpy + gpsd uniquement

---

## Démarrage rapide (SITL)

```bash
# Stack complète (mosquitto + sim-engine + 3 nœuds simulés + master)
docker compose up -d --build

# Lancer un scénario 3 minutes
python scripts/run_scenario.py scripts/scenarios/orbit_uhf.yaml --duration 180 --no-nodes

# Évaluer
python scripts/eval.py \
  --scenario scripts/scenarios/orbit_uhf.yaml \
  --log-dir logs/run-<latest>/

# WebUI
open http://localhost:8080
```

---

## Performances actuelles

### SITL Provence — orbit_uhf (Friis, 1 UAV orbit 800m + 2 nœuds sol, 433 MHz)

| Modèle | CEP50 | CEP90 | Notes |
|--------|-------|-------|-------|
| GridLikelihoodModel | **28m** | **60m** | Analytique Friis, convergence immédiate |
| UNet ITM Provence v3 | 920m | 1411m | Mismatch propagation (entraîné ITM ≠ Friis simu) |

### POWDER — données réelles (462 MHz, campus University of Utah)

| Modèle | CEP50 | CEP90 | Notes |
|--------|-------|-------|-------|
| Friis absolu | 211m | 806m | Baseline |
| **UNet + terrain** | **69m** | **97m** | 3× mieux que Friis, in-domain |

---

## Structure

```
containers/
  master/          — localisation + inférence + WebUI (FastAPI + Leaflet)
  node-hitl/       — nœud SDR physique (SoapySDR, ARM)
  node-sitl/       — nœud simulé (orbite UAV, trajectoire GPX)
  sim-engine/      — moteur propagation RF (ITM / Friis)
shared/
  comm/            — abstraction transport (MQTT, WebSocket, Zenoh)
  messages/        — schémas Pydantic inter-services
  geo/             — preprocessing géospatial (SRTM, OSM, Copernicus)
  models/          — wrappers UNet / GridLikelihood / DiffusionModel
training/
  generate_dataset.py
  train_unet.py
scripts/
  run_scenario.py  — lance un scénario SITL
  eval.py          — calcule CEP50/CEP90/convergence
  fetch_terrain.py — télécharge SRTM/Copernicus/OSM
  export_onnx.py   — exporte checkpoint PyTorch → ONNX
deploy/            — units systemd + scripts install bare metal
```

---

## Déploiement bare metal

```bash
# Master x86 (calcul + WebUI)
sudo bash deploy/install-master.sh

# Nœud ARM (RPi 4/5, CM4, Jetson)
sudo bash deploy/install-node.sh
```

Voir `docs/deploy-arm.md` et `CLAUDE.md` pour la documentation complète.
