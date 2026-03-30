"""
config.py
---------
Parámetros globales del pipeline de detección de oil slicks.
Modificar este archivo para adaptar el sistema a una nueva región o período.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Rutas (se pueden sobreescribir via variables de entorno o argumentos CLI)
# ---------------------------------------------------------------------------

# Raíz del proyecto en Google Drive (solo relevante en Colab)
DRIVE_ROOT = Path("/content/drive/MyDrive/TAMI_YPF_Global")

PATHS = {
    "images_dir":    DRIVE_ROOT / "1_Imagenes_SAR",
    "metadata_dir":  DRIVE_ROOT / "2_Metadatos",
    "models_dir":    DRIVE_ROOT / "Modelos",
    "evidence_dir":  DRIVE_ROOT / "Evidencias_Visuales",
    "master_csv":    DRIVE_ROOT / "2_Metadatos" / "Registro_Maestro_Golfo.csv",
    "images_zip":    DRIVE_ROOT / "1_Imagenes_SAR_Local.zip",
    # Directorios temporales en la SSD local de Colab
    "local_tifs":    Path("/content/TIFs_Temporales"),
    "local_infer":   Path("/content/TIFs_Inferencia"),
}

GEE_PROJECT = "tamiypf"   # <-- Reemplazar con el nombre de tu proyecto GCP


# ---------------------------------------------------------------------------
# Área de interés (AOI) — Golfo San Matías, Playa Bonita
# ---------------------------------------------------------------------------

AOI = {
    "lon_min": -63.35,
    "lon_max": -62.75,
    "lat_min": -41.40,
    "lat_max": -40.95,
    # Resolución de la grilla: ~4 km entre centros de tile
    "grid_step_deg": 0.036,
    # Radio del buffer alrededor de cada punto central (metros)
    "tile_buffer_m": 2000,
    # Dimensiones del tile descargado (píxeles)
    "tile_px": "400x400",
}


# ---------------------------------------------------------------------------
# Filtros de adquisición
# ---------------------------------------------------------------------------

ACQUISITION = {
    "date_start": "2024-01-01",
    "date_end":   "2024-12-31",
    # Rango de viento válido (ERA5, m/s)
    "wind_min_ms": 3.0,
    "wind_max_ms": 10.0,
    # Umbral de elevación (ETOPO1): solo píxeles bajo el mar
    "elevation_threshold_m": -5,
    # Modo y polarización de Sentinel-1
    "s1_mode":         "IW",
    "s1_polarization": "VV",
    # Máximo de imágenes por punto de la grilla
    "max_images_per_tile": 100,
    # Hilos de descarga concurrente
    "download_workers": 5,
}


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------

PREPROCESSING = {
    # Filtro de Lee
    "lee_kernel_size": 5,
    # Filtro de Bradley
    "bradley_window_ratio": 9,
    "bradley_T": 23,
    # Morfología (radio en píxeles)
    "morph_radius": 3,
    # Normalización SAR (rango dB → [0, 255])
    "sar_db_min": -25.0,
    "sar_db_max":   0.0,
    # Umbral de píxeles vacíos (NaN) para descartar un tile
    "max_nodata_fraction": 0.05,
    # Radio del suavizado focal (panorámica de contexto, metros)
    "panoramic_smooth_radius_m": 50,
    "panoramic_px": 1200,
    "panoramic_buffer_m": 7500,
}


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

MODEL = {
    "input_dim":       16,
    "num_classes":      2,
    # Nombres de los archivos del modelo (dentro de PATHS["models_dir"])
    "weights_file":    "feature_net_model.pth",
    "scaler_file":     "scaler.joblib",
    # Umbral mínimo de confianza para registrar una detección positiva
    "confidence_threshold": 60.0,
    # Etiquetas de clase
    "label_positive": "Slick Petroleo",
    "label_negative": "Falsa Alarma",
}


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

VIZ = {
    "map_center": [-41.15, -63.0],
    "map_zoom":   10,
    # Umbral mínimo de confianza para incluir un punto en el heatmap
    "heatmap_min_confidence": 60.0,
    "output_map_filename": "Mapa_Calor_Resultados.html",
}
