# lunchfork — Terrain Data

## Sources

| Source | Resolution | Coverage | Auth required |
|--------|-----------|----------|---------------|
| SRTM GL1 | 30m (1 arc-sec) | Global | No (ESA STEP mirror) |
| Copernicus GLO-30 | 25m | Global | No (AWS Open Data) |
| IGN RGE Alti | 1m | France | Géoportail account |
| OSM Buildings | vector | Global | No (Overpass API) |
| Corine Land Cover | 100m | Europe | No (EEA WCS) |

## Fetching terrain data

```bash
# SRTM — global baseline
python scripts/fetch_terrain.py --source srtm --bbox 4.0 43.0 6.5 44.5

# Copernicus — better European coverage
python scripts/fetch_terrain.py --source copernicus --bbox 4.0 43.0 6.5 44.5

# OSM buildings — for urban scenarios
python scripts/fetch_terrain.py --source osm-buildings --bbox 4.5 43.4 5.2 43.8

# Corine land cover — for vegetation attenuation
python scripts/fetch_terrain.py --source corine --bbox 4.0 43.0 6.5 44.5

# All sources at once
python scripts/fetch_terrain.py --source all --bbox 4.0 43.0 6.5 44.5
```

## Cache structure

```
data/terrain/
├── srtm/
│   ├── N43E004.hgt
│   ├── N43E005.hgt
│   └── index.json
├── copernicus/
│   ├── Copernicus_DSM_COG_10_N43_00_E004_00_DEM.tif
│   └── index.json
├── osm/
│   └── buildings_43.4000_4.5000_43.8000_5.2000.geojson
└── corine/
    └── clc2018_43.400_4.500_43.800_5.200.tif
```

## GeoPreprocessor

The `GeoPreprocessor` class (in `shared/geo`) assembles a `[3, H, W]` conditioning tensor:

- **Channel 0**: Normalised DEM (SRTM/Copernicus, bilinear interpolated to grid)
- **Channel 1**: Building height (from OSM footprints + height tags) — zero until Phase 3
- **Channel 2**: Vegetation attenuation (from Corine land cover) — zero until Phase 3

Priority order for DEM: IGN RGE (1m) > Copernicus (25m) > SRTM (30m)

## Storage estimates (50×50 km region)

| Source | Size |
|--------|------|
| SRTM tiles | ~8 MB |
| Copernicus tiles | ~15 MB |
| OSM buildings | ~5–30 MB (depends on urban density) |
| Corine | ~2 MB |
| **Total** | **< 60 MB** |

This fits comfortably on an RPi SD card for embedded deployment.
