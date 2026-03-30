"""
acquisition.py
--------------
Descarga concurrente de imágenes Sentinel-1 desde Google Earth Engine (GEE).

Flujo
-----
1. Se genera una grilla de puntos sobre el AOI.
2. Se filtran los puntos terrestres con ETOPO1 (elevación < -5 m).
3. Para cada punto, se descarga cada imagen de 2024 si el viento ERA5
   está dentro del rango válido (3–10 m/s).
4. Las descargas se realizan en paralelo con ThreadPoolExecutor.
5. Los metadatos se persisten en el CSV maestro tras cada cuadrante.

Nota sobre autenticación
------------------------
Antes de importar este módulo, el entorno debe haber ejecutado:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)
"""

from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

import ee  # type: ignore

from src.config import AOI, ACQUISITION, PATHS


# ---------------------------------------------------------------------------
# Generación de grilla
# ---------------------------------------------------------------------------

def build_water_grid(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    step: float,
    elevation_threshold: float = -5.0,
) -> list[list[float]]:
    """
    Crea una grilla de puntos sobre el AOI y retiene solo los acuáticos.

    Aplica la máscara topográfica ETOPO1: se descartan todos los puntos con
    elevación ≥ `elevation_threshold` metros (por defecto -5 m).

    Retorna
    -------
    list[list[float]]  Lista de [lon, lat] para cada punto marino.
    """
    features = [
        ee.Feature(ee.Geometry.Point([lon, lat]))
        for lon in np.arange(lon_min, lon_max, step)
        for lat in np.arange(lat_min, lat_max, step)
    ]

    elev_image  = ee.Image("NOAA/NGDC/ETOPO1").select("bedrock")
    with_elev   = elev_image.reduceRegions(
        collection=ee.FeatureCollection(features),
        reducer=ee.Reducer.mean().setOutputs(["elevacion"]),
        scale=1000,
    )
    water_points = (
        with_elev
        .filter(ee.Filter.lt("elevacion", elevation_threshold))
        .geometry()
        .coordinates()
        .getInfo()
    )
    print(f"✅ Grilla lista: {len(water_points)} cuadrantes marinos.")
    return water_points


# ---------------------------------------------------------------------------
# Worker de descarga (ejecutado en cada hilo)
# ---------------------------------------------------------------------------

def _download_tile(
    i: int,
    image_list: ee.List,
    point: ee.Geometry,
    aoi: ee.Geometry,
    lon: float,
    lat: float,
    local_dir: Path,
    date_start: str,
    date_end: str,
    wind_min: float,
    wind_max: float,
    tile_px: str,
    processed_files: set[str],
    results: list[dict[str, Any]],
    lock: threading.Lock,
) -> bool:
    """Descarga una sola imagen SAR para el punto (lon, lat)."""
    try:
        image     = ee.Image(image_list.get(i))
        fecha_str = image.date().format("YYYY-MM-dd").getInfo()
        fecha_obj = datetime.strptime(fecha_str, "%Y-%m-%d")
        year_str  = fecha_obj.strftime("%Y")
        month_str = fecha_obj.strftime("%m")
        filename  = f"SAR_{fecha_str}_Lon{lon:.3f}_Lat{lat:.3f}.tif"

        with lock:
            if filename in processed_files:
                return True

        # --- Filtro de viento (ERA5) ---
        fecha_ee  = image.date()
        era5_col  = (
            ee.ImageCollection("ECMWF/ERA5/HOURLY")
            .filterBounds(point)
            .filterDate(fecha_ee, fecha_ee.advance(1, "day"))
        )
        wind_ms = -1.0
        try:
            if era5_col.size().getInfo() > 0:
                winds   = (
                    era5_col.mean()
                    .reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=10000)
                    .getInfo()
                )
                u = winds.get("u_component_of_wind_10m")
                v = winds.get("v_component_of_wind_10m")
                if u is not None and v is not None:
                    wind_ms = math.sqrt(u ** 2 + v ** 2)
        except Exception:
            pass

        status = "DESCARTADO_CLIMA"

        if wind_min <= wind_ms <= wind_max:
            rel_dir = Path(year_str) / month_str
            dest    = local_dir / rel_dir
            dest.mkdir(parents=True, exist_ok=True)

            try:
                smoothed = image.select("VV").focal_median(radius=30, units="meters")
                url = smoothed.getDownloadURL({
                    "dimensions": tile_px,
                    "crs":        "EPSG:4326",
                    "region":     aoi,
                    "format":     "GEO_TIFF",
                })
                resp = requests.get(url, timeout=20)
                if resp.status_code == 200:
                    (dest / filename).write_bytes(resp.content)
                    status = "OK"
                    print(f"   ⚡ [OK] {fecha_str} | Viento: {wind_ms:.1f} m/s")
                else:
                    status = "ERROR_API"
            except Exception:
                status = "ERROR_RED"

        record = {
            "Archivo":        filename,
            "Ruta_Relativa":  str(Path(year_str) / month_str),
            "Fecha":          fecha_str,
            "Año":            year_str,
            "Mes":            month_str,
            "Lon":            lon,
            "Lat":            lat,
            "Viento_ms":      round(wind_ms, 2),
            "Confiabilidad":  "ALTA" if wind_min <= wind_ms <= wind_max else "BAJA",
            "Estado_Descarga":status,
            "Prediccion_IA":  "Pendiente",
            "Confianza_IA":   0.0,
        }
        with lock:
            results.append(record)
            processed_files.add(filename)

        return True

    except Exception:
        return False


# ---------------------------------------------------------------------------
# Bucle principal de descarga
# ---------------------------------------------------------------------------

def download_sar_dataset(
    water_coords: list[list[float]],
    local_dir: Path,
    master_csv: Path,
    date_start: str | None = None,
    date_end:   str | None = None,
    wind_min:   float | None = None,
    wind_max:   float | None = None,
    tile_px:    str | None = None,
    buffer_m:   int = 2000,
    workers:    int = 5,
    max_images: int = 100,
) -> pd.DataFrame:
    """
    Descarga todo el dataset SAR sobre la grilla de puntos marinos.

    Soporta checkpointing: si ya existe `master_csv`, retoma desde donde
    quedó sin volver a descargar archivos ya procesados.

    Parámetros
    ----------
    water_coords : list  Salida de `build_water_grid`.
    local_dir    : Path  Directorio local donde guardar los TIF.
    master_csv   : Path  Ruta del CSV maestro (checkpoint).
    date_start / date_end : str  Rango temporal (ISO 8601).
    wind_min / wind_max   : float  Rango de viento válido (m/s).
    tile_px      : str   Dimensiones del tile ("400x400").
    buffer_m     : int   Radio del AOI alrededor de cada punto (metros).
    workers      : int   Hilos paralelos de descarga.
    max_images   : int   Límite de imágenes por punto de grilla.

    Retorna
    -------
    pd.DataFrame  DataFrame maestro con todos los metadatos.
    """
    # Valores por defecto desde config
    date_start = date_start or ACQUISITION["date_start"]
    date_end   = date_end   or ACQUISITION["date_end"]
    wind_min   = wind_min   if wind_min is not None else ACQUISITION["wind_min_ms"]
    wind_max   = wind_max   if wind_max is not None else ACQUISITION["wind_max_ms"]
    tile_px    = tile_px    or AOI["tile_px"]

    # Checkpoint
    if master_csv.exists():
        df_existing = pd.read_csv(master_csv)
        results          = df_existing.to_dict("records")
        processed_files  = set(df_existing["Archivo"].tolist())
        print(f"🔄 Retomando desde checkpoint: {len(results)} registros existentes.")
    else:
        results, processed_files = [], set()

    lock = threading.Lock()

    for idx, coords in enumerate(water_coords):
        lon, lat = coords[0], coords[1]
        point    = ee.Geometry.Point([lon, lat])
        aoi      = point.buffer(buffer_m).bounds()

        print(f"\n📍 Cuadrante {idx + 1}/{len(water_coords)} "
              f"(Lon: {lon:.3f}, Lat: {lat:.3f})")

        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi)
            .filterDate(date_start, date_end)
            .filter(ee.Filter.eq("instrumentMode", ACQUISITION["s1_mode"]))
            .filter(ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                ACQUISITION["s1_polarization"],
            ))
        )

        try:
            n = collection.size().getInfo()
        except Exception:
            continue

        if n == 0:
            continue

        image_list = collection.toList(max_images)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _download_tile,
                    i, image_list, point, aoi, lon, lat,
                    local_dir, date_start, date_end,
                    wind_min, wind_max, tile_px,
                    processed_files, results, lock,
                )
                for i in range(n)
            ]
            for _ in as_completed(futures):
                pass

        # Guardar checkpoint al finalizar cada cuadrante
        df = pd.DataFrame(results)
        master_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(master_csv, index=False)

    print(f"\n🎉 Descarga completada. Total registros: {len(results)}.")

    # Comprimir TIFs y subir a Drive para persistirlos entre sesiones de Colab
    zip_path = master_csv.parent.parent / "1_Imagenes_SAR_Local.zip"
    print(f"\n📦 Comprimiendo TIFs en {zip_path} ...")
    import shutil as _shutil
    _shutil.make_archive(
        base_name=str(zip_path.with_suffix("")),
        format="zip",
        root_dir=str(local_dir.parent),
        base_dir=local_dir.name,
    )
    print(f"✅ ZIP guardado en Drive: {zip_path}")

    return pd.DataFrame(results)
