# CLAUDE.md — lunchfork

> Document de référence architecture pour Claude Code et tout contributeur.
> Projet personnel sous Monadic Labs. Ne pas publier sans révision.

---

## Vue d'ensemble

**lunchfork** est un système de localisation passive de mobiles RF (VHF/UHF, non-coopératifs) basé sur :
- Collecte RSSI distribuée (nœuds SDR hétérogènes : UAV, sol, véhicules)
- Reconstruction de radio map par diffusion conditionnelle (RadioDiff-Loc inspired)
- Tracking multi-cibles par filtre particulaire
- WebUI cartographique temps réel

**Objectif de précision cible :**
- UHF (400–900 MHz) : 20–50m avec 2–3 orbites UAV
- VHF (100–300 MHz) : 80–200m avec 2–3 orbites UAV
- Approche coarse-to-fine sur grille + super-résolution ×4 conditionnée MNT

---

## Environnements de déploiement

| Env | Usage | Runtime |
|---|---|---|
| `dev` | Développement, SITL complet | Docker Compose, x86 |
| `prod-master` | Calculateur central terrain | Bare metal x86, avec ou sans GPU |
| `prod-node` | Nœuds capteurs mobiles | Bare metal ARM (RPi 4/5, CM4, Jetson Nano) |

### Séparation des rôles — règle absolue

```
Nœuds ARM (node-hitl)
  ✅ SDR + FFT + clustering fréquentiel
  ✅ GPS + horodatage
  ✅ Publication RSSI via CommTransport
  ❌ PAS d'inférence ML
  ❌ PAS de modèle embarqué
  ❌ PAS de filtre particulaire

Master x86 (master)
  ✅ Toute l'inférence (diffusion + SR)
  ✅ Filtre particulaire
  ✅ Geo preprocessing
  ✅ WebUI + API
  ✅ ONNX Runtime CPU si pas de GPU
  ✅ PyTorch ou ONNX Runtime CUDA si GPU dispo
```

### Implications master x86

- **Sans GPU** : ONNX Runtime CPU — latence 3–10s par inférence selon résolution, suffisant pour loitering UAV
- **Avec GPU** : PyTorch CUDA ou ONNX Runtime CUDA — latence < 1s, confortable
- `docker compose` pour dev et SITL
- En prod terrain : bare metal, services `systemd`, pas de Docker (overhead inutile)
- L'entraînement des modèles se fait offline sur machine x86/GPU dédiée — hors scope runtime

### Implications nœuds ARM

- Aucune dépendance ML (pas de PyTorch, pas d'ONNX Runtime)
- Dépendances : SoapySDR, numpy (FFT), gpsd, lib comm (paho-mqtt ou zenoh-python)
- Empreinte mémoire minimale — tourne confortablement sur RPi 4 2GB
- `systemd` pour gestion des services

---

## Philosophie de développement

- **Simulation d'abord** : tout nouveau composant est validé en SITL avant tout test HITL
- **Interfaces stables** : les contrats entre services sont versionnés et documentés ici
- **Abstraction comm** : aucun composant n'importe directement `paho-mqtt` ou `zenoh` — toujours via `CommTransport`
- **ONNX en prod master** : modèles entraînés offline en PyTorch, exportés `.onnx` — ONNX Runtime CPU/CUDA selon hardware master
- **Reproductibilité** : seeds fixées, configs versionnées, résultats loggés pour comparaison
- **Nœuds sans ML** : les nœuds ARM n'embarquent aucun modèle — FFT + RSSI + GPS uniquement

---

## Structure du dépôt

```
lunchfork/
├── CLAUDE.md
├── docker-compose.yml              # stack dev complète
├── docker-compose.sitl.yml         # override SITL
├── docker-compose.hitl.yml         # override HITL
│
├── containers/
│   ├── master/                     # localisation + inference + webui
│   │   ├── main.py
│   │   ├── pipeline/               # clustering, sliding window, PF, zoom
│   │   ├── static/                 # WebUI (Leaflet, JS vanilla)
│   │   └── requirements.txt
│   ├── node-hitl/                  # nœud SDR physique
│   ├── node-sitl/                  # nœud simulé
│   └── sim-engine/                 # moteur propagation RF
│
├── shared/                         # lib Python partagée (installée en editable)
│   ├── comm/                       # abstraction transport
│   ├── messages/                   # schémas Pydantic inter-services
│   ├── geo/                        # preprocessing géospatial
│   └── models/                     # wrappers inférence ONNX/PyTorch
│
├── data/
│   ├── terrain/                    # cache MNT + bâtiments (gitignored)
│   ├── datasets/                   # datasets synthétiques (gitignored)
│   └── checkpoints/                # .onnx + .pt (gitignored)
│
├── training/                       # offline, x86/GPU uniquement
│   ├── generate_dataset.py
│   ├── train_diffusion.py
│   └── train_superres.py
│
├── deploy/                         # bare metal prod
│   ├── systemd/                    # units systemd
│   ├── install.sh
│   └── config.prod.env             # template config prod
│
├── scripts/
│   ├── fetch_terrain.py
│   ├── export_onnx.py
│   ├── run_scenario.py
│   └── eval.py
│
└── docs/
    ├── architecture.md
    ├── messages.md
    ├── terrain.md
    └── deploy-arm.md
```

---

## Services

### `master` — Localisation + Inférence + WebUI

**Responsabilités :**
- Réception des mesures RSSI via `CommTransport`
- Clustering fréquentiel → canaux détectés
- Fenêtre glissante adaptative par canal
- Pipeline : geo preprocessing → diffusion coarse → zoom → diffusion fine → SR×4
- Filtre particulaire multi-cibles
- WebUI Leaflet + API REST debug + WebSocket push

**Stack :**
- Python 3.11
- **ONNX Runtime CPU** (prod x86 sans GPU) ou **ONNX Runtime CUDA / PyTorch CUDA** (prod x86 avec GPU) — switché via `INFERENCE_BACKEND`
- FastAPI + Uvicorn
- Leaflet.js servi statiquement
- `shared/comm`, `shared/geo`, `shared/models`

**Ports :**
- `8080` — WebUI + API REST
- `8081` — WebSocket events

**Variables d'environnement :**
```
COMM_TRANSPORT=mqtt|zenoh|websocket
COMM_BROKER_URL=mqtt://localhost:1883

INFERENCE_BACKEND=onnx-cpu|onnx-cuda|pytorch-cuda   # selon hardware master
MODEL_DIFFUSION_PATH=data/checkpoints/diffusion_vhf_v1.onnx
MODEL_SR_PATH=data/checkpoints/sr_vhf_v1.onnx

TERRAIN_CACHE_DIR=data/terrain
TERRAIN_PREFER=copernicus             # srtm|copernicus|ign-rge

GRID_COARSE_SIZE=256
GRID_FINE_SIZE=256
SR_FACTOR=4
ZOOM_TRIGGER_STD_M=500

PF_N_PARTICLES=500                    # réduire à 200 sur RPi 4
PF_MOTION_MODEL=constant_velocity
PF_ADAPTIVE_MOTION=true
SLIDING_WINDOW_SEC=30
SLIDING_WINDOW_ADAPTIVE=true

FREQ_CLUSTER_BW_HZ=25000
FREQ_CLUSTER_THRESHOLD_DBM=-90

LOG_LEVEL=INFO
```

**Points d'attention :**
- Inférence dans `ThreadPoolExecutor` — ne jamais bloquer la boucle asyncio
- Sans GPU : `PF_N_PARTICLES=500`, latence 3–10s par update — acceptable pour loitering UAV (orbite ~2 min)
- Avec GPU : latence < 1s, `PF_N_PARTICLES` peut monter à 2000+
- Le zoom coarse→fine se déclenche quand `pf.position_std_m < ZOOM_TRIGGER_STD_M`
- FastAPI sert l'API debug et les fichiers statiques — pas de logique métier dans les routes

---

### `node-hitl` — Nœud SDR physique

**Responsabilités :**
- Pilotage SDR via SoapySDR (RTL-SDR, HackRF, LimeSDR)
- FFT fenêtre glissante → clustering fréquentiel → RSSI par canal
- Horodatage GPS (gpsd ou NMEA direct)
- Publication `RssiMessage` via `CommTransport`
- Mode replay : rejouer un fichier IQ enregistré (SigMF ou IQ binaire brut)

**Variables d'environnement :**
```
COMM_TRANSPORT=mqtt|zenoh|websocket
COMM_BROKER_URL=...
NODE_ID=node-001
NODE_TYPE=ground|uav|vehicle

SDR_DRIVER=rtlsdr|hackrf|limesdr
SDR_FREQ_CENTER_HZ=433920000
SDR_SAMPLE_RATE=2048000
SDR_GAIN=40
FFT_SIZE=2048
FFT_OVERLAP=0.5
FREQ_CLUSTER_THRESHOLD_DBM=-90

GPS_SOURCE=gpsd|nmea|static
GPS_DEVICE=/dev/ttyACM0
GPS_STATIC_LAT=43.530
GPS_STATIC_LON=5.450
GPS_STATIC_ALT_M=200

REPLAY_FILE=                          # chemin IQ si mode replay
LOG_LEVEL=INFO
```

**Points d'attention :**
- `NODE_ID` unique dans la flotte — hostname ou UUID fixé dans config
- Clustering fréquentiel côté nœud — minimise le débit réseau vers master
- Mode `REPLAY_FILE` : rejoue des captures terrain sans hardware — utile pour validation
- Sur RPi + RTL-SDR : fixer `cpu_freq` au max dans `/boot/config.txt`

---

### `node-sitl` — Nœud simulé

**Responsabilités :**
- Simuler un nœud SDR sans hardware
- Interroger `sim-engine` pour le RSSI à sa position courante
- Simuler une trajectoire (statique, orbite UAV, route, replay GPX)
- Publier des `RssiMessage` au format **identique** à `node-hitl`

**Variables d'environnement :**
```
COMM_TRANSPORT=mqtt|zenoh|websocket
COMM_BROKER_URL=...
NODE_ID=sim-node-001
NODE_TYPE=ground|uav|vehicle

TRAJECTORY_TYPE=static|orbit|route|gps_replay
TRAJECTORY_ORBIT_LAT=43.535
TRAJECTORY_ORBIT_LON=5.455
TRAJECTORY_ORBIT_RADIUS_M=800
TRAJECTORY_ORBIT_ALT_M=120
TRAJECTORY_ORBIT_PERIOD_SEC=120
TRAJECTORY_ORBIT_HELIX=false          # true = orbite hélicoïdale (alt variable)
TRAJECTORY_GPS_FILE=                  # GPX/CSV si gps_replay

SIMENGINE_URL=http://sim-engine:9000
UPDATE_RATE_HZ=1
LOG_LEVEL=INFO
```

**Points d'attention :**
- Le master ne distingue pas `node-sitl` de `node-hitl` — format de message identique
- Plusieurs instances simultanées possibles (flotte simulée) — différencier par `NODE_ID`
- `gps_replay` permet de rejouer une trajectoire terrain réelle pour valider le pipeline

---

### `sim-engine` — Moteur de simulation propagation RF

**Responsabilités :**
- Calculer RSSI reçu pour un couple (émetteur, capteur)
- Générer des radio maps ground truth pour l'entraînement
- Gérer plusieurs émetteurs simultanément
- Exposer une API REST interne

**Stack :**
- Python 3.11, FastAPI + Uvicorn
- **ITM/Longley-Rice** — moteur principal (CPU, VHF/UHF outdoor)
- **SIONNA** optionnel (GPU, ray-tracing, génération dataset offline uniquement)
- `shared/geo`

**API REST :**
```
POST /rssi
  { freq_hz, emitter: {lat, lon, alt_m}, sensor: {lat, lon, alt_m} }
  → { rssi_dbm, is_nlos, path_loss_db }

POST /radiomap
  { freq_hz, emitter: {lat, lon, alt_m},
    bbox: {lat_min, lon_min, lat_max, lon_max}, resolution_px }
  → { radiomap: [[float]], bbox, resolution_m_per_px, crs }

POST   /emitter      { id, lat, lon, alt_m, freq_hz, power_dbm } → { id }
DELETE /emitter/{id}
GET    /emitters     → [{ id, lat, lon, alt_m, freq_hz, power_dbm }]
GET    /health
```

**Variables d'environnement :**
```
PROPAGATION_MODEL=itm|sionna|friis
TERRAIN_CACHE_DIR=data/terrain
SIONNA_GPU=false
ITM_CLIMATE=5                         # 5 = continental temperate
ITM_PERMITTIVITY=15
ITM_CONDUCTIVITY=0.005
LOG_LEVEL=INFO
```

**Points d'attention :**
- `friis` = espace libre, utile pour tests rapides sans terrain
- SIONNA uniquement en génération offline dataset (`SIONNA_GPU=true`)
- Ne démarre pas en mode HITL pur

---

## Couche de communication — `shared/comm`

### Interface

```python
from abc import ABC, abstractmethod
from typing import Callable, Awaitable
from shared.messages import BaseMessage

class CommTransport(ABC):
    @abstractmethod
    async def connect(self) -> None: ...
    @abstractmethod
    async def disconnect(self) -> None: ...
    @abstractmethod
    async def publish(self, topic: str, message: BaseMessage) -> None: ...
    @abstractmethod
    async def subscribe(
        self,
        topic_pattern: str,
        callback: Callable[[str, BaseMessage], Awaitable[None]],
    ) -> None: ...
    @abstractmethod
    async def healthcheck(self) -> bool: ...

def get_transport() -> CommTransport:
    """Lit COMM_TRANSPORT depuis env, retourne l'implémentation."""
    ...
```

### Implémentations

| `COMM_TRANSPORT` | Classe | Broker | Notes |
|---|---|---|---|
| `mqtt` | `MqttTransport` | Mosquitto | défaut, robuste |
| `zenoh` | `ZenohTransport` | non (P2P) | sans broker, terrain déconnecté |
| `websocket` | `WebSocketTransport` | non | dev only |

Ajouter un transport = implémenter `CommTransport` + enregistrer dans `get_transport()`. Aucune autre modification.

### Topics

```
rssi/{node_id}                        # RssiMessage — nœud → master
node/status/{node_id}                 # NodeStatus  — nœud → master
loc/target/{target_id}                # TargetLocation — master → UI
loc/radiomap/{freq_hz}                # RadioMapUpdate — master → UI
sim/emitter/{emitter_id}              # EmitterState — sim → log
```

---

## Messages — `shared/messages`

Pydantic v2, sérialisés JSON.

```python
class NodePosition(BaseModel):
    lat: float
    lon: float
    alt_m: float
    accuracy_m: float | None = None

class FreqChannel(BaseModel):
    center_hz: int
    bandwidth_hz: int
    label: str | None = None

class RssiMessage(BaseModel):
    version: str = "1.0"
    node_id: str
    node_type: Literal["ground", "uav", "vehicle"]
    timestamp_utc: datetime
    position: NodePosition
    freq_channel: FreqChannel
    rssi_dbm: float
    snr_db: float | None = None
    is_simulated: bool = False

class NodeStatus(BaseModel):
    version: str = "1.0"
    node_id: str
    timestamp_utc: datetime
    node_type: Literal["ground", "uav", "vehicle"]
    position: NodePosition
    sdr_ok: bool
    gps_ok: bool
    comm_ok: bool
    rssi_rate_hz: float

class TargetLocation(BaseModel):
    version: str = "1.0"
    target_id: str
    timestamp_utc: datetime
    position: NodePosition
    uncertainty_m: float
    covariance: list[list[float]]     # 2×2 (ou 3×3 phase 6)
    n_particles: int
    freq_channel: FreqChannel
    track_state: Literal["init", "confirmed", "lost"]

class RadioMapUpdate(BaseModel):
    version: str = "1.0"
    timestamp_utc: datetime
    freq_channel: FreqChannel
    bbox: dict                        # {lat_min, lon_min, lat_max, lon_max}
    resolution_m_per_px: float
    data_b64: str                     # float32 [H,W] base64
```

---

## Pipeline d'inférence — `master`

### Clustering fréquentiel

```python
class FreqClusterer:
    """
    Regroupe les RssiMessage par canal fréquentiel.
    Crée dynamiquement de nouveaux canaux à la détection.
    Fusionne les canaux trop proches (< FREQ_CLUSTER_BW_HZ).
    """
    def push(self, msg: RssiMessage) -> str: ...      # retourne channel_id
    def get_channels(self) -> list[FreqChannel]: ...
```

### Fenêtre glissante

```python
class SlidingWindow:
    """
    Buffer circulaire par channel_id.
    Taille adaptative : réduite si le PF détecte une accélération.
    """
    def push(self, channel_id: str, msg: RssiMessage) -> None: ...
    def get_snapshot(self, channel_id: str) -> list[RssiMessage]: ...
    def adapt(self, channel_id: str, tracker_state: TrackerState) -> None: ...
```

### Geo Preprocessor — `shared/geo`

```python
class GeoPreprocessor:
    """
    Tenseur conditioning [3, H, W] float32 :
      canal 0 : MNT normalisé
      canal 1 : hauteur bâtiments normalisée
      canal 2 : atténuation végétation

    Priorité sources MNT :
      1. RGE Alti IGN 1m (France)
      2. Copernicus DEM 25m (Europe)
      3. SRTM 30m (global)
    Bâtiments : BD TOPO IGN → OSM fallback
    Végétation : Corine Land Cover
    CRS : UTM zone locale (auto depuis bbox)
    """
    def get_conditioning_tensor(
        self, bbox: BBox, resolution_px: int,
    ) -> np.ndarray: ...           # [3, H, W]

    def get_mnt_hires(
        self, bbox: BBox, resolution_px: int,
    ) -> np.ndarray: ...           # [1, H, W] pour SR
```

### Wrappers modèles — `shared/models`

```python
class InferenceBackend(ABC):
    """Abstraction PyTorch / ONNX Runtime."""
    @abstractmethod
    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...

def get_backend(model_path: str) -> InferenceBackend:
    backend = os.environ["INFERENCE_BACKEND"]
    if backend == "onnx-cpu":
        return OnnxBackend(model_path, providers=["CPUExecutionProvider"])
    if backend == "onnx-cuda":
        return OnnxBackend(model_path, providers=["CUDAExecutionProvider"])
    if backend == "pytorch-cuda":
        return TorchBackend(model_path, device="cuda")
    return TorchBackend(model_path, device="cpu")  # dev fallback

class DiffusionModel:
    def infer(
        self,
        conditioning: np.ndarray,         # [3, H, W]
        sparse_rssi: list[tuple[float, float, float]],  # (x_px, y_px, rssi_norm)
        n_steps: int = 50,
    ) -> np.ndarray: ...                  # [1, H, W]

class SuperResolutionModel:
    def upscale(
        self,
        radiomap_lr: np.ndarray,          # [1, H, W]
        mnt_hr: np.ndarray,               # [1, H*SR, W*SR]
    ) -> np.ndarray: ...                  # [1, H*SR, W*SR]
```

### Boucle pipeline coarse → fine → SR

```python
snapshot = sliding_window.get_snapshot(channel_id)

# --- coarse ---
cond_c = geo.get_conditioning_tensor(zone_large, GRID_COARSE_SIZE)
rm_c   = diffusion.infer(cond_c, to_sparse(snapshot, zone_large))
pf.update(channel_id, rm_c, zone_large)

# --- zoom si convergence suffisante ---
if pf.position_std_m(channel_id) < ZOOM_TRIGGER_STD_M:
    bbox_fine = pf.get_confidence_bbox(channel_id, sigma=2.0)
    cond_f  = geo.get_conditioning_tensor(bbox_fine, GRID_FINE_SIZE)
    rm_f    = diffusion.infer(cond_f, to_sparse(snapshot, bbox_fine))
    mnt_hr  = geo.get_mnt_hires(bbox_fine, GRID_FINE_SIZE * SR_FACTOR)
    rm_hr   = sr_model.upscale(rm_f, mnt_hr)
    pf.update(channel_id, rm_hr, bbox_fine)
```

### Filtre particulaire

```python
class ParticleFilter:
    """
    Multi-cibles. Une instance par channel_id.
    Likelihood : interpolation bilinéaire radio_map(x_particule).
    Resampling : systematic resampling si ESS < N/2.
    Modèle mouvement : CV par défaut, switch CT si virage détecté (adaptatif).
    """
    def update(self, channel_id: str, radiomap: np.ndarray, bbox: BBox) -> None: ...
    def get_targets(self, channel_id: str) -> list[TargetLocation]: ...
    def get_confidence_bbox(self, channel_id: str, sigma: float) -> BBox: ...
    def position_std_m(self, channel_id: str) -> float: ...
```

---

## WebUI

Single-page app, `containers/master/static/`. Leaflet.js + JS vanilla.

**Affichage :**
- Fond OSM (tiles cachées offline si déploiement déconnecté)
- Nœuds actifs : icônes par type + couleur statut
- Cibles : marker + ellipse incertitude 1-sigma + trajectoire (50 pts)
- Radio map heatmap overlay (opacité slider)
- Panel nœuds : liste + statut + débit RSSI
- Panel spectre : FFT temps réel nœud sélectionné (canvas, WebSocket)

**API REST debug :**
```
GET  /api/targets
GET  /api/nodes
GET  /api/radiomap/{freq_hz}
GET  /api/health
POST /api/scenario              # dev only
```

**WebSocket `/ws` port 8081** — push JSON vers UI.

---

## Scénarios SITL

```yaml
# scripts/scenarios/orbit_uhf.yaml
name: "UAV orbit UHF 433MHz"
duration_sec: 600
terrain_bbox: [4.5, 43.4, 5.2, 43.8]

emitters:
  - id: target-1
    lat: 43.530
    lon: 5.450
    alt_m: 5
    freq_hz: 433920000
    power_dbm: 10
    trajectory: static

nodes:
  - id: uav-1
    type: uav
    trajectory:
      type: orbit
      center_lat: 43.535
      center_lon: 5.455
      radius_m: 800
      alt_m: 120
      period_sec: 120
      helix: false

  - id: ground-1
    type: ground
    lat: 43.510
    lon: 5.430
    alt_m: 180

eval:
  ground_truth_emitter: target-1
  metrics: [rmse_m, cep50, cep90, convergence_time_sec]
  convergence_threshold_m: 200
```

```bash
python scripts/run_scenario.py scripts/scenarios/orbit_uhf.yaml
python scripts/eval.py --scenario scripts/scenarios/orbit_uhf.yaml \
  --log-dir logs/run-$(date +%Y%m%d)/ --output results/
```

---

## Export ONNX

```bash
python scripts/export_onnx.py \
  --model diffusion \
  --checkpoint data/checkpoints/diffusion_vhf_v1.pt \
  --output data/checkpoints/diffusion_vhf_v1.onnx \
  --opset 17

python scripts/export_onnx.py \
  --model superres \
  --checkpoint data/checkpoints/sr_vhf_v1.pt \
  --output data/checkpoints/sr_vhf_v1.onnx \
  --opset 17

# Benchmark ARM (via qemu ou directement sur RPi)
python scripts/export_onnx.py --benchmark \
  --checkpoint data/checkpoints/diffusion_vhf_v1.onnx
```

---

## Données terrain

```bash
python scripts/fetch_terrain.py --source srtm       --bbox 4.0 43.0 6.5 44.5
python scripts/fetch_terrain.py --source copernicus --bbox 4.0 43.0 6.5 44.5
python scripts/fetch_terrain.py --source osm-buildings --bbox 4.0 43.0 6.5 44.5
python scripts/fetch_terrain.py --source corine     --bbox 4.0 43.0 6.5 44.5
```

Cache : `data/terrain/{srtm,copernicus,osm,corine}/` + `index.json`.
CRS : UTM zone locale, géré transparentement par `rasterio` + `pyproj`.

**Taille 50×50 km :** SRTM + Copernicus + OSM + Corine ≈ **< 60 MB** — embarquable RPi.

---

## Déploiement bare metal prod

### Master x86

```bash
sudo bash deploy/install-master.sh
# - installe python3.11, onnxruntime (CPU ou CUDA selon détection GPU)
# - copie units systemd → /etc/systemd/system/
# - active lunchfork-master, lunchfork-broker (Mosquitto)
```

Config : `deploy/config.master.env` → `/etc/lunchfork/config.env`.

**Sans GPU :** `INFERENCE_BACKEND=onnx-cpu` — suffisant pour validation et opération standard.
**Avec GPU :** `INFERENCE_BACKEND=onnx-cuda` ou `pytorch-cuda` — recommandé si disponible.

### Nœuds ARM (RPi 4/5, CM4, Jetson Nano)

```bash
sudo bash deploy/install-node.sh
# - installe python3.11, soapysdr, gpsd, paho-mqtt
# - PAS de pytorch, PAS d'onnxruntime
# - copie lunchfork-node-hitl.service
```

Config : `deploy/config.node.env` → `/etc/lunchfork/config.env`.

**Empreinte nœud ARM :** ~150MB RAM, 0% GPU, tourne sur RPi 4 2GB sans problème.

---

## Génération dataset + entraînement

```bash
# Dataset synthétique
python training/generate_dataset.py \
  --n-scenes 10000 --freq-range 100e6 900e6 \
  --terrain-zones france,benelux \
  --propagation-model itm \
  --output data/datasets/vhf_uhf_outdoor_v1/

# Entraînement diffusion (fine-tune depuis RadioDiff-Loc)
python training/train_diffusion.py \
  --dataset data/datasets/vhf_uhf_outdoor_v1/ \
  --base-checkpoint radiodiff-loc-pretrained.pt \
  --output data/checkpoints/diffusion_vhf_v1.pt \
  --epochs 200 --batch-size 16

# Entraînement SR×4
python training/train_superres.py \
  --dataset data/datasets/vhf_uhf_outdoor_v1/ \
  --sr-factor 4 \
  --output data/checkpoints/sr_vhf_v1.pt \
  --epochs 100 --batch-size 32
```

---

## Conventions de code

- Python 3.11+, type hints partout, pas de `Any` non justifié
- Pydantic v2 pour tous les modèles de données inter-services
- `asyncio` pour toute I/O — inférence dans `ThreadPoolExecutor` dédié
- `numpy` dans le chemin critique (pas `torch` en prod ARM)
- Logging : `structlog` JSON, corrélation par `node_id` / `channel_id`
- Tests : `pytest` + `pytest-asyncio`
- Linting : `ruff` / formatting : `black`
- Dépendances minimales — stdlib si suffisant

---

## Roadmap

### Phase 1 — Infrastructure + SITL minimal
- [ ] Structure dépôt + Docker Compose
- [ ] `shared/comm` : interface + `MqttTransport`
- [ ] `shared/messages` : schémas Pydantic
- [ ] `sim-engine` : Friis simple (sans terrain)
- [ ] `node-sitl` : trajectoire statique, publication mesures
- [ ] `master` : réception + fenêtre glissante + log

### Phase 2 — Simulation réaliste
- [ ] ITM dans `sim-engine`
- [ ] `shared/geo` : MNT SRTM + rasterisation OSM
- [ ] `node-sitl` : orbite UAV + helix
- [ ] Génération premier dataset VHF/UHF

### Phase 3 — Pipeline ML
- [ ] Entraînement diffusion sur dataset synthétique
- [ ] `shared/models` : `DiffusionModel` + backends ONNX/PyTorch
- [ ] `master` : pipeline coarse
- [ ] PF basique (random walk)
- [ ] Évaluation CEP50/CEP90 SITL

### Phase 4 — Pipeline complet
- [ ] Zoom coarse→fine
- [ ] `SuperResolutionModel` + entraînement
- [ ] PF constant velocity + adaptatif
- [ ] WebUI : carte + cibles + nœuds + heatmap
- [ ] Export ONNX + benchmark ARM

### Phase 5 — HITL
- [ ] `node-hitl` : SoapySDR + FFT + clustering
- [ ] GPS via gpsd + mode replay IQ
- [ ] Tests terrain

### Phase 6 — Extensions
- [ ] `ZenohTransport` (sans broker)
- [ ] Extension 2.5D altitude émetteur
- [ ] Orbite hélicoïdale dans PF
- [ ] Segmentation post radio map (multi-émetteurs même canal)
- [ ] Tiles OSM offline WebUI

---

## Questions ouvertes / décisions différées

- **Modèle diffusion** : RadioDiff-Loc as-is pour valider (Phase 3), fine-tune VHF/UHF en Phase 4. Re-train from scratch seulement si résultats Phase 3 insuffisants.
- **Altitude émetteur** : ignorée jusqu'en Phase 6 (2D). L'orbite hélicoïdale UAV sera le premier cas 2.5D.
- **Tiles OSM offline WebUI** : non prioritaire. À adresser en Phase 6 si déploiement totalement déconnecté.
- **Auth WebUI** : pas d'auth pour l'instant (réseau privé local). À évaluer si exposition réseau public.
- **GPU master** : ONNX CPU suffit pour valider le pipeline. GPU accélère le cycle d'itération en dev et devient utile si on monte `PF_N_PARTICLES` ou la résolution de grille.

