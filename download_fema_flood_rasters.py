#!/usr/bin/env python3
"""Download FEMA NFHL flood hazard polygons state-by-state and rasterize to single-digit classes.

Example:
  python download_fema_flood_rasters.py --states CA TX --resolution 10 --output-dir output
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import array_bounds, from_origin
from rasterio.warp import Resampling, reproject, transform_bounds

# FEMA NFHL service layer for flood hazard polygons.
# Service root: https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer
FEMA_LAYER_URL = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"

# TIGER/Line US states polygons (full-resolution boundaries; includes territories).
STATE_BOUNDARY_URL = "https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip"

STATE_ABBRS = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}
TERRITORY_ABBRS = {"AS", "GU", "MP", "PR", "VI"}


# Detailed FEMA categorical mapping.
# 0 is reserved for nodata/background.
ZONE_SUBTY_TO_CODE: Dict[str, int] = {
    "AREA OF MINIMAL FLOOD HAZARD": 1,
    "0.2 PCT ANNUAL CHANCE FLOOD HAZARD": 2,
    "FLOODWAY": 12,
    "REGULATORY FLOODWAY": 13,
    "ADMINISTRATIVE FLOODWAY": 14,
    "CHANNEL": 15,
    "COASTAL FLOODPLAIN": 16,
    "FUTURE CONDITIONS 1% ANNUAL CHANCE FLOOD HAZARD": 17,
    "AREA WITH REDUCED FLOOD RISK DUE TO LEVEE": 18,
    "LEVEE PROTECTED AREA": 19,
}

# Fallback mapping by FLD_ZONE field.
FLD_ZONE_TO_CODE: Dict[str, int] = {
    "D": 3,
    "A": 4,
    "AE": 5,
    "AH": 6,
    "AO": 7,
    "A99": 8,
    "AR": 9,
    "V": 10,
    "VE": 11,
}


@dataclass
class StateInfo:
    abbr: str
    name: str
    fips: str
    geometry: object


@dataclass
class RasterGrid:
    crs: str
    transform: object
    width: int
    height: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_run_log(
    log_path: Path,
    state: str,
    status: str,
    message: str,
    elapsed_seconds: float,
    feature_count: int = 0,
    width: int = 0,
    height: int = 0,
) -> None:
    header = [
        "timestamp_utc",
        "state",
        "status",
        "message",
        "elapsed_seconds",
        "feature_count",
        "width",
        "height",
    ]
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(
            [
                utc_now_iso(),
                state,
                status,
                message,
                f"{elapsed_seconds:.2f}",
                feature_count,
                width,
                height,
            ]
        )


def write_state_stats_json(output_path: Path, raster: np.ndarray) -> None:
    vals, counts = np.unique(raster, return_counts=True)
    total = int(raster.size)
    payload = {
        "total_pixels": total,
        "class_pixel_counts": {str(int(v)): int(c) for v, c in zip(vals, counts)},
        "class_pixel_percent": {str(int(v)): round((int(c) / total) * 100.0, 6) for v, c in zip(vals, counts)},
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def request_json(
    session: requests.Session,
    url: str,
    params: Dict[str, object],
    timeout: int,
    max_retries: int,
    backoff_seconds: float,
) -> dict:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            sleep_for = backoff_seconds * (2**attempt) + random.uniform(0, 0.25)
            time.sleep(sleep_for)
    raise RuntimeError(f"Request failed after {max_retries + 1} attempts: {last_exc}")


def request_response(
    session: requests.Session,
    url: str,
    timeout: int,
    max_retries: int,
    backoff_seconds: float,
    params: Optional[Dict[str, object]] = None,
    stream: bool = False,
) -> requests.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout, stream=stream)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            sleep_for = backoff_seconds * (2**attempt) + random.uniform(0, 0.25)
            time.sleep(sleep_for)
    raise RuntimeError(f"Request failed after {max_retries + 1} attempts: {last_exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and rasterize FEMA NFHL data by state")
    parser.add_argument(
        "--states",
        nargs="+",
        help="State abbreviations (e.g. CA TX NY). If omitted, all states + DC are processed.",
    )
    parser.add_argument(
        "--scope",
        choices=["states", "states+territories"],
        default="states+territories",
        help="Scope used when --states is omitted (default: states+territories)",
    )
    parser.add_argument(
        "--source-mode",
        choices=["hybrid", "state-package", "rest"],
        default="hybrid",
        help="FEMA source mode: hybrid (state package first, REST fallback), state-package, or rest.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Base output directory for per-state GeoJSON and GeoTIFF files",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=10.0,
        help="Raster resolution in output CRS units (default 10)",
    )
    parser.add_argument(
        "--crs",
        default="EPSG:3857",
        help="Output raster CRS (default: EPSG:3857)",
    )
    parser.add_argument(
        "--max-record-count",
        type=int,
        default=200,
        help="Features per FEMA object-id chunk query (default 200)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="HTTP timeout seconds",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.1,
        help="Sleep between paginated FEMA API requests",
    )
    parser.add_argument(
        "--save-vector",
        action="store_true",
        help="Save downloaded vector polygons for each state",
    )
    parser.add_argument(
        "--vector-format",
        choices=["gpkg", "geojson"],
        default="gpkg",
        help="Vector output format when --save-vector is used (default: gpkg)",
    )
    parser.add_argument(
        "--water-mask",
        help=(
            "Optional water mask path (raster or vector). "
            "Mask water pixels are forced to 0 nodata in output flood raster."
        ),
    )
    parser.add_argument(
        "--shared-grid",
        action="store_true",
        help=(
            "Use one shared raster grid for all outputs in this run "
            "(same transform/width/height for every state)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip states whose output raster already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and state selection without downloading FEMA data.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries for HTTP/API requests (default 5).",
    )
    parser.add_argument(
        "--backoff-seconds",
        type=float,
        default=1.0,
        help="Base retry backoff seconds; exponential with jitter (default 1.0).",
    )
    return parser.parse_args()


def load_states(selected_states: Optional[Sequence[str]], scope: str) -> List[StateInfo]:
    gdf = gpd.read_file(STATE_BOUNDARY_URL)
    # Exclude non-CONUS states/territories if needed; keep all states + DC by default.
    gdf = gdf[gdf["STUSPS"].notna()].copy()

    if selected_states:
        wanted = {s.upper() for s in selected_states}
        gdf = gdf[gdf["STUSPS"].str.upper().isin(wanted)]
        missing = wanted - set(gdf["STUSPS"].str.upper())
        if missing:
            raise ValueError(f"State code(s) not found: {', '.join(sorted(missing))}")
    else:
        if scope == "states":
            allowed = STATE_ABBRS
        else:
            allowed = STATE_ABBRS | TERRITORY_ABBRS
        gdf = gdf[gdf["STUSPS"].str.upper().isin(allowed)]

    gdf = gdf.to_crs("EPSG:4326")
    states = [
        StateInfo(
            abbr=row["STUSPS"],
            name=row["NAME"],
            fips=str(row["STATEFP"]).zfill(2),
            geometry=row.geometry,
        )
        for _, row in gdf.iterrows()
    ]
    return states


def normalize_fema_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    fields = ["FLD_ZONE", "ZONE_SUBTY", "SFHA_TF", "STATIC_BFE"]
    if gdf.crs is None:
        # FEMA NFHL source layers are generally NAD83 geographic if CRS is missing.
        gdf = gdf.set_crs("EPSG:4269", allow_override=True)
    gdf = gdf.to_crs("EPSG:4326")
    for col in fields:
        if col not in gdf.columns:
            gdf[col] = None
    return gdf[fields + ["geometry"]].copy()


def fetch_fema_features_from_state_package(
    state: StateInfo,
    session: requests.Session,
    timeout: int,
    max_retries: int,
    backoff_seconds: float,
) -> gpd.GeoDataFrame:
    search_url = (
        "https://msc.fema.gov/portal/advanceSearch"
        f"?affiliate=fema&query&selstate={state.fips}&selcounty={state.fips}001"
        f"&selcommunity={state.fips}001C&searchedCid={state.fips}001C&method=search"
    )
    response = request_response(
        session=session,
        url=search_url,
        timeout=timeout,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )
    try:
        search_payload = response.json()
    except ValueError:
        search_payload = json.loads(response.text)

    product_path = (
        search_payload.get("EFFECTIVE", {})
        .get("NFHL_STATE_DATA", [{}])[0]
        .get("product_FILE_PATH")
    )
    if not product_path:
        raise RuntimeError(f"No state package product path found for {state.abbr}")

    package_url = f"https://hazards.fema.gov/nfhlv2/output/State/{product_path}"
    package_response = request_response(
        session=session,
        url=package_url,
        timeout=timeout,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )

    with tempfile.TemporaryDirectory(prefix=f"fema_{state.abbr}_") as tmpdir:
        with zipfile.ZipFile(io.BytesIO(package_response.content)) as zf:
            zf.extractall(tmpdir)

        root = Path(tmpdir)

        # Prefer geodatabase layer for fidelity.
        for gdb_path in root.rglob("*.gdb"):
            # First try pyogrio (more reliable for OpenFileGDB in many envs).
            try:
                import pyogrio
                layer_rows = pyogrio.list_layers(gdb_path)
                layer_names = [str(row[0]) for row in layer_rows]
                target = next((ly for ly in layer_names if ly.lower() == "s_fld_haz_ar"), None)
                if target is None:
                    target = next((ly for ly in layer_names if "fld_haz_ar" in ly.lower()), None)
                if target:
                    gdf = gpd.read_file(gdb_path, layer=target, engine="pyogrio")
                    return normalize_fema_gdf(gdf)
            except Exception:
                pass

            # Fallback to fiona if pyogrio route fails.
            try:
                import fiona
                layers = fiona.listlayers(gdb_path)
                target = next((ly for ly in layers if ly.lower() == "s_fld_haz_ar"), None)
                if target is None:
                    target = next((ly for ly in layers if "fld_haz_ar" in ly.lower()), None)
                if target:
                    gdf = gpd.read_file(gdb_path, layer=target)
                    return normalize_fema_gdf(gdf)
            except Exception:
                continue

        # Fallback to shapefile if no readable GDB layer.
        for shp_path in root.rglob("*.shp"):
            if shp_path.name.lower() == "s_fld_haz_ar.shp" or "fld_haz_ar" in shp_path.name.lower():
                gdf = gpd.read_file(shp_path)
                return normalize_fema_gdf(gdf)

    raise RuntimeError(f"Could not find S_Fld_Haz_Ar in FEMA package for {state.abbr}")


def fetch_fema_features_for_state(
    state: StateInfo,
    session: requests.Session,
    timeout: int,
    max_record_count: int,
    sleep_seconds: float,
    max_retries: int,
    backoff_seconds: float,
) -> gpd.GeoDataFrame:
    fields = ["FLD_ZONE", "ZONE_SUBTY", "SFHA_TF", "STATIC_BFE"]
    all_features: List[dict] = []

    minx, miny, maxx, maxy = state.geometry.bounds

    ids_params = {
        "f": "json",
        "where": "1=1",
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "returnIdsOnly": "true",
    }
    ids_payload = request_json(
        session=session,
        url=FEMA_LAYER_URL,
        params=ids_params,
        timeout=timeout,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )
    object_ids: List[int] = ids_payload.get("objectIds", []) or []

    def fetch_chunk_geojson(chunk_ids: List[int]) -> List[dict]:
        where_in = ",".join(str(oid) for oid in chunk_ids)
        params = {
            "f": "geojson",
            "where": f"OBJECTID IN ({where_in})",
            "outFields": ",".join(fields),
            "outSR": 4326,
        }
        try:
            payload = request_json(
                session=session,
                url=FEMA_LAYER_URL,
                params=params,
                timeout=timeout,
                max_retries=max_retries,
                backoff_seconds=backoff_seconds,
            )
        except RuntimeError as exc:
            if len(chunk_ids) > 1:
                mid = len(chunk_ids) // 2
                return fetch_chunk_geojson(chunk_ids[:mid]) + fetch_chunk_geojson(chunk_ids[mid:])
            if len(chunk_ids) == 1:
                print(
                    f"  Warning: skipping OBJECTID {chunk_ids[0]} due to repeated FEMA/API errors ({exc})",
                    file=sys.stderr,
                )
                return []
            raise
        if "features" not in payload:
            raise RuntimeError(
                f"Unexpected FEMA response for {state.abbr}: {str(payload)[:300]}"
            )
        return payload["features"]

    chunk_size = max(1, int(max_record_count))
    for idx in range(0, len(object_ids), chunk_size):
        chunk_ids = object_ids[idx : idx + chunk_size]
        all_features.extend(fetch_chunk_geojson(chunk_ids))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not all_features:
        return gpd.GeoDataFrame(columns=fields + ["geometry"], geometry="geometry", crs="EPSG:4326")

    gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")
    for col in fields:
        if col not in gdf.columns:
            gdf[col] = None
    return gdf


def map_feature_to_digit(zone_subty: Optional[str], fld_zone: Optional[str], sfha_tf: Optional[str]) -> int:
    def clean_text(value: object) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        text = str(value).strip()
        return text if text else None

    zone_subty = clean_text(zone_subty)
    fld_zone = clean_text(fld_zone)
    sfha_tf = clean_text(sfha_tf)

    if zone_subty:
        key = zone_subty.upper()
        if key in ZONE_SUBTY_TO_CODE:
            return ZONE_SUBTY_TO_CODE[key]

    if fld_zone:
        fz = fld_zone.upper()
        if fz == "X" or fz.startswith("X "):
            return 23
        if fz in FLD_ZONE_TO_CODE:
            return FLD_ZONE_TO_CODE[fz]
        # Some records include composite values like "AE FLOODWAY".
        for prefix, code in FLD_ZONE_TO_CODE.items():
            if fz.startswith(prefix):
                return code

    # Fallback with SFHA flag: T means special flood hazard area.
    if sfha_tf:
        sfha = sfha_tf.upper()
        if sfha == "T":
            return 20
        if sfha == "F":
            return 21

    return 22


def build_water_mask_array(
    water_mask_path: Path,
    out_height: int,
    out_width: int,
    out_transform,
    out_crs: str,
) -> np.ndarray:
    suffix = water_mask_path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        with rasterio.open(water_mask_path) as src:
            if src.crs is None:
                raise ValueError(f"Water mask raster has no CRS: {water_mask_path}")

            dst_bounds = array_bounds(out_height, out_width, out_transform)
            src_bounds = transform_bounds(out_crs, src.crs, *dst_bounds, densify_pts=21)
            src_window = rasterio.windows.from_bounds(*src_bounds, transform=src.transform)
            src_window = src_window.round_offsets().round_lengths()

            fill_value = src.nodata if src.nodata is not None else 0
            src_arr = src.read(1, window=src_window, boundless=True, fill_value=fill_value)
            src_nodata = src.nodata
            src_valid = np.ones(src_arr.shape, dtype=bool)
            if src_nodata is not None:
                src_valid = src_arr != src_nodata
            src_water = (src_arr != 0) & src_valid

            dst = np.zeros((out_height, out_width), dtype=np.uint8)
            reproject(
                source=src_water.astype(np.uint8),
                destination=dst,
                src_transform=src.window_transform(src_window),
                src_crs=src.crs,
                dst_transform=out_transform,
                dst_crs=out_crs,
                resampling=Resampling.nearest,
            )
            return dst.astype(bool)

    # Treat all other paths as vector data readable by GeoPandas/OGR.
    water_gdf = gpd.read_file(water_mask_path)
    if water_gdf.empty:
        return np.zeros((out_height, out_width), dtype=bool)
    if water_gdf.crs is None:
        raise ValueError(f"Water mask vector has no CRS: {water_mask_path}")
    water_proj = water_gdf.to_crs(out_crs)
    shapes = (
        (geom, 1)
        for geom in water_proj.geometry
        if geom is not None and not geom.is_empty
    )
    mask = rasterize(
        shapes=shapes,
        out_shape=(out_height, out_width),
        transform=out_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    return mask.astype(bool)


def build_shared_grid(states: Sequence[StateInfo], resolution_m: float, out_crs: str) -> RasterGrid:
    state_geoms = gpd.GeoSeries([s.geometry for s in states], crs="EPSG:4326").to_crs(out_crs)
    minx, miny, maxx, maxy = state_geoms.total_bounds
    width = max(1, int(math.ceil((maxx - minx) / resolution_m)))
    height = max(1, int(math.ceil((maxy - miny) / resolution_m)))
    transform = from_origin(minx, maxy, resolution_m, resolution_m)
    pixel_count = width * height
    if pixel_count > 1_000_000_000:
        raise RuntimeError(
            f"Shared grid is too large ({pixel_count:,} pixels). "
            "Reduce scope, increase resolution size, or disable --shared-grid."
        )
    return RasterGrid(crs=out_crs, transform=transform, width=width, height=height)


def rasterize_state(
    gdf_4326: gpd.GeoDataFrame,
    state: StateInfo,
    output_path: Path,
    resolution_m: float,
    output_crs: str,
    water_mask_path: Optional[Path] = None,
    output_grid: Optional[RasterGrid] = None,
) -> Tuple[int, int, int, np.ndarray]:
    if gdf_4326.empty:
        raise ValueError(f"No features to rasterize for {state.abbr}")

    gdf_proj = gdf_4326.to_crs(output_crs)

    values = [
        map_feature_to_digit(
            row.get("ZONE_SUBTY"),
            row.get("FLD_ZONE"),
            row.get("SFHA_TF"),
        )
        for _, row in gdf_proj.iterrows()
    ]

    if output_grid is None:
        minx, miny, maxx, maxy = gdf_proj.total_bounds
        width = max(1, int(math.ceil((maxx - minx) / resolution_m)))
        height = max(1, int(math.ceil((maxy - miny) / resolution_m)))
        transform = from_origin(minx, maxy, resolution_m, resolution_m)
    else:
        width = output_grid.width
        height = output_grid.height
        transform = output_grid.transform

    shapes = ((geom, val) for geom, val in zip(gdf_proj.geometry, values) if geom is not None and not geom.is_empty)
    raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )

    if water_mask_path is not None:
        water_mask = build_water_mask_array(
            water_mask_path=water_mask_path,
            out_height=height,
            out_width=width,
            out_transform=transform,
            out_crs=output_crs,
        )
        raster[water_mask] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="COG",
        height=height,
        width=width,
        count=1,
        dtype=rasterio.uint8,
        crs=output_crs,
        transform=transform,
        compress="LZW",
        BIGTIFF="IF_SAFER",
        blocksize=512,
        overview_resampling="NEAREST",
        nodata=0,
    ) as dst:
        dst.write(raster, 1)

    feature_count = int(len(gdf_proj))
    return width, height, feature_count, raster


def write_mapping_json(output_dir: Path) -> None:
    mapping_path = output_dir / "class_mapping.json"
    payload = {
        "description": "Detailed FEMA categorical flood mapping. One code per distinct map meaning where possible. 0 is nodata/background.",
        "classes": {
            "0": "NoData / background",
            "1": "Zone X - area of minimal flood hazard",
            "2": "Zone X - 0.2% annual chance flood hazard",
            "3": "Area of undetermined flood hazard",
            "4": "Zone A",
            "5": "Zone AE",
            "6": "Zone AH",
            "7": "Zone AO",
            "8": "Zone A99",
            "9": "Zone AR",
            "10": "Zone V",
            "11": "Zone VE",
            "12": "Floodway",
            "13": "Regulatory floodway",
            "14": "Administrative floodway",
            "15": "Channel",
            "16": "Coastal floodplain",
            "17": "Future conditions 1% annual chance flood hazard",
            "18": "Area with reduced flood risk due to levee",
            "19": "Levee protected area",
            "20": "SFHA true fallback",
            "21": "SFHA false fallback",
            "22": "Unknown / unmapped fallback",
            "23": "Zone X - unresolved subtype",
        },
        "ZONE_SUBTY_TO_CODE": ZONE_SUBTY_TO_CODE,
        "FLD_ZONE_TO_CODE": FLD_ZONE_TO_CODE,
        "fallback": {
            "x_unresolved": 23,
            "sfha_true": 20,
            "sfha_false": 21,
            "default": 22,
        },
    }
    mapping_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resolution <= 0:
        print("--resolution must be > 0", file=sys.stderr)
        return 1
    if args.max_retries < 0:
        print("--max-retries must be >= 0", file=sys.stderr)
        return 1
    if args.backoff_seconds < 0:
        print("--backoff-seconds must be >= 0", file=sys.stderr)
        return 1
    water_mask_path = Path(args.water_mask).expanduser().resolve() if args.water_mask else None
    if water_mask_path and not water_mask_path.exists():
        print(f"Water mask path does not exist: {water_mask_path}", file=sys.stderr)
        return 1

    try:
        states = load_states(args.states, args.scope)
    except Exception as exc:
        print(f"Failed to load state boundaries: {exc}", file=sys.stderr)
        return 1

    write_mapping_json(output_dir)
    run_log_path = output_dir / "run_log.csv"

    session = requests.Session()
    session.headers.update({"User-Agent": "fema-rasterizer/1.0"})
    shared_grid = build_shared_grid(states, args.resolution, args.crs) if args.shared_grid else None
    if shared_grid is not None:
        print(
            f"Using shared grid: {shared_grid.width}x{shared_grid.height}, "
            f"CRS {shared_grid.crs}, resolution {args.resolution}"
        )
    if args.dry_run:
        print("Dry run only. No FEMA downloads will be performed.")
        print("Selected states:")
        print("  " + ", ".join(s.abbr for s in states))
        print(f"Run log path: {run_log_path}")
        return 0

    print(f"Processing {len(states)} state(s)")
    failures: List[Tuple[str, str]] = []

    for i, state in enumerate(states, start=1):
        print(f"[{i}/{len(states)}] {state.abbr} - {state.name}")
        started = time.time()
        state_dir = output_dir / state.abbr
        tif_path = state_dir / f"{state.abbr}_fema_flood_classes.tif"

        if args.resume and tif_path.exists():
            print(f"  Skipping {state.abbr}; raster already exists")
            append_run_log(
                run_log_path,
                state=state.abbr,
                status="SKIPPED",
                message="Raster exists and --resume enabled",
                elapsed_seconds=time.time() - started,
            )
            continue

        try:
            if args.source_mode == "state-package":
                gdf = fetch_fema_features_from_state_package(
                    state=state,
                    session=session,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    backoff_seconds=args.backoff_seconds,
                )
            elif args.source_mode == "rest":
                gdf = fetch_fema_features_for_state(
                    state=state,
                    session=session,
                    timeout=args.timeout,
                    max_record_count=args.max_record_count,
                    sleep_seconds=args.sleep_seconds,
                    max_retries=args.max_retries,
                    backoff_seconds=args.backoff_seconds,
                )
            else:
                try:
                    print("  Source: FEMA state package")
                    gdf = fetch_fema_features_from_state_package(
                        state=state,
                        session=session,
                        timeout=args.timeout,
                        max_retries=args.max_retries,
                        backoff_seconds=args.backoff_seconds,
                    )
                except Exception as package_exc:
                    print(f"  State package failed; falling back to REST ({package_exc})", file=sys.stderr)
                    gdf = fetch_fema_features_for_state(
                        state=state,
                        session=session,
                        timeout=args.timeout,
                        max_record_count=args.max_record_count,
                        sleep_seconds=args.sleep_seconds,
                        max_retries=args.max_retries,
                        backoff_seconds=args.backoff_seconds,
                    )

            if gdf.empty:
                print(f"  No FEMA flood features found for {state.abbr}; skipping raster")
                append_run_log(
                    run_log_path,
                    state=state.abbr,
                    status="EMPTY",
                    message="No FEMA features found",
                    elapsed_seconds=time.time() - started,
                )
                continue

            state_dir.mkdir(parents=True, exist_ok=True)

            if args.save_vector:
                if args.vector_format == "gpkg":
                    vec_path = state_dir / f"{state.abbr}_fema_flood.gpkg"
                    gdf.to_file(vec_path, driver="GPKG", layer="fema_flood")
                else:
                    vec_path = state_dir / f"{state.abbr}_fema_flood.geojson"
                    gdf.to_file(vec_path, driver="GeoJSON")
                print(f"  Saved vector: {vec_path}")

            width, height, feature_count, raster = rasterize_state(
                gdf_4326=gdf,
                state=state,
                output_path=tif_path,
                resolution_m=args.resolution,
                output_crs=args.crs,
                water_mask_path=water_mask_path,
                output_grid=shared_grid,
            )
            stats_path = state_dir / f"{state.abbr}_fema_flood_stats.json"
            write_state_stats_json(stats_path, raster)
            print(f"  Rasterized {feature_count} features -> {tif_path} ({width}x{height})")
            append_run_log(
                run_log_path,
                state=state.abbr,
                status="SUCCESS",
                message="Completed",
                elapsed_seconds=time.time() - started,
                feature_count=feature_count,
                width=width,
                height=height,
            )

        except Exception as exc:
            failures.append((state.abbr, str(exc)))
            print(f"  ERROR: {exc}", file=sys.stderr)
            append_run_log(
                run_log_path,
                state=state.abbr,
                status="FAILED",
                message=str(exc),
                elapsed_seconds=time.time() - started,
            )

    if failures:
        print("\nCompleted with errors:", file=sys.stderr)
        for abbr, msg in failures:
            print(f"  {abbr}: {msg}", file=sys.stderr)
        return 2

    print("\nAll requested states processed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
