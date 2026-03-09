# FEMA Flood Rasterizer

Download FEMA National Flood Hazard Layer (NFHL) polygons by state and rasterize them into single-digit flood classes.

## What this creates

For each state code you request:

- `output/<STATE>/<STATE>_fema_flood_classes.tif` (COG, output CRS, `uint8`)
- optional `output/<STATE>/<STATE>_fema_flood.gpkg` (downloaded vector data; default)
- `output/class_mapping.json` (the digit mapping used)

The default class logic uses `ZONE_SUBTY` first, then `FLD_ZONE`, then `SFHA_TF` fallback.
Examples:

- `FLOODWAY -> 7`
- `LEVEE PROTECTED AREA -> 9`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Process selected states:

```bash
python download_fema_flood_rasters.py --states CA TX FL --resolution 10 --output-dir output --save-vector
```

Process all states + DC:

```bash
python download_fema_flood_rasters.py --scope states --resolution 10 --output-dir output
```

Process all states + US territories:

```bash
python download_fema_flood_rasters.py --scope states+territories --resolution 10 --output-dir output
```

## Key options

- `--states`: list of state abbreviations (omit for all)
- `--scope`: when `--states` is omitted, use `states` or `states+territories` (default)
- `--source-mode`: `hybrid` (default), `state-package`, or `rest`
- `--crs`: output raster CRS (default `EPSG:3857`)
- `--resolution`: output raster pixel size in output CRS units (default `10`)
- `--save-vector`: also save state GeoJSON polygons
- `--max-record-count`: FEMA object-id chunk size (default `200`)
- `--water-mask`: optional raster/vector water mask; masked pixels are forced to `0` nodata
- `--vector-format`: `gpkg` (default) or `geojson` when using `--save-vector`
- `--shared-grid`: use one common grid (same transform/width/height) for all rasters in the run
- `--resume`: skip states where raster already exists
- `--dry-run`: validate configuration and selected states without downloading data
- `--max-retries`: request retries for FEMA/API calls (default `5`)
- `--backoff-seconds`: exponential backoff base seconds for retries (default `1.0`)

Run outputs include:

- `run_log.csv` in the output root (state-level status/timing)
- `<STATE>_fema_flood_stats.json` per state (pixel counts by class)

## CI

GitHub Actions workflow: `.github/workflows/ci.yml`

- Lint: `ruff check download_fema_flood_rasters.py --select E9,F63,F7,F82`
- Smoke test: `python download_fema_flood_rasters.py --states RI --dry-run --output-dir ci_output`

Example with water mask clipping:

```bash
python download_fema_flood_rasters.py --states RI --resolution 3 --output-dir output_3m --water-mask /path/to/water_mask.tif
```

Example using FEMA state packages first (with REST fallback):

```bash
python download_fema_flood_rasters.py --states RI --source-mode hybrid --output-dir output_hybrid
```

## Customize class digits

Edit these dictionaries in `download_fema_flood_rasters.py`:

- `ZONE_SUBTY_TO_CODE`
- `FLD_ZONE_TO_CODE`

All class values should remain 0-9 if you need strict single-digit encoding.
