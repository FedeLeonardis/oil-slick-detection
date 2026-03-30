"""
visualization.py
----------------
Mapa de calor interactivo y auditoría visual ("La Lupa") para las detecciones
de oil slicks.

El mapa replica el estilo del notebook original:
- Fondo oscuro CartoDB Dark Matter
- HeatMap ponderado por confianza
- Marcadores circulares cyan con popup básico
- Popup con coordenada, total detecciones, fechas y confianza promedio

La Lupa puede activarse de dos formas:
1. Desde el mapa en Colab: llamar register_lupa_callback() antes de mostrar
   el mapa. Agrega un botón en cada popup que dispara extract_evidence()
   directamente desde el kernel via output.eval_js / output_callbacks.
2. Manual desde el notebook: llamar extract_evidence() con la coordenada.
"""

from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import pandas as pd
import requests
import rasterio
from folium.plugins import HeatMap
from PIL import Image

import ee  # type: ignore

from src.config import PREPROCESSING, VIZ


# ---------------------------------------------------------------------------
# Mapa de calor
# ---------------------------------------------------------------------------

def build_heatmap(
    master_csv: Path | pd.DataFrame,
    output_path: Path,
    min_confidence: float | None = None,
    map_center: list[float] | None = None,
    zoom: int | None = None
) -> folium.Map:
    """
    Genera el mapa de calor interactivo con las detecciones positivas.

    Parámetros
    ----------
    master_csv         : Path | DataFrame  CSV maestro o DataFrame cargado.
    output_path        : Path  Ruta donde guardar el HTML.
    min_confidence     : float  Umbral mínimo de confianza (default del config).
    map_center         : [lat, lon]  Centro inicial del mapa.
    zoom               : int  Nivel de zoom inicial.
    enable_lupa_button : bool  Incluir botón "La Lupa" en los popups.
                         Solo funciona en Colab con register_lupa_callback().

    Retorna
    -------
    folium.Map
    """
    min_confidence = min_confidence or VIZ["heatmap_min_confidence"]
    map_center     = map_center     or VIZ["map_center"]
    zoom           = zoom           or VIZ["map_zoom"]

    if isinstance(master_csv, Path):
        df = pd.read_csv(master_csv)
    else:
        df = master_csv.copy()

    positives = df[
        (df["Prediccion_IA"] == "Slick Petroleo") &
        (df["Confianza_IA"]  >= min_confidence)
    ].copy()

    print(f"🗺️ Generando Mapa de Calor...")

    m = folium.Map(
        location=map_center,
        zoom_start=zoom,
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="© OpenStreetMap contributors © CARTO",
    )

    # --- HeatMap ---
    heat_data = [
        [row["Lat"], row["Lon"], row["Confianza_IA"] / 100.0]
        for _, row in positives.iterrows()
    ]
    if heat_data:
        HeatMap(heat_data, radius=20, blur=15, min_opacity=0.4).add_to(m)

    # --- Marcadores por coordenada única ---
    grouped = positives.groupby(["Lat", "Lon"])
    for (lat, lon), group in grouped:
        n_detections   = len(group)
        avg_confidence = group["Confianza_IA"].mean()
        fechas         = ", ".join(sorted(group["Fecha"].tolist()))

        lupa_btn = ""
        

        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; width: 220px;">
            <b>📍 Coordenada:</b> {lat:.3f}, {lon:.3f}<br>
            <b>🚨 Total Detecciones:</b> {n_detections}<br>
            <b>📅 Fechas:</b> {fechas}<br>
            <b>🎯 Confianza Promedio:</b> {avg_confidence:.1f}%
            {lupa_btn}
        </div>
        """
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color="cyan",
            fill=True,
            fill_color="cyan",
            fill_opacity=0.8,
            weight=3,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(m)


    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"✅ ¡Mapa generado con éxito en: {output_path}")

    return m


# ---------------------------------------------------------------------------
# Registro del callback de La Lupa para Colab
# ---------------------------------------------------------------------------

def register_lupa_callback(
    master_csv: Path | pd.DataFrame,
    tifs_dir: Path,
    output_base_dir: Path,
    lee_filter_fn,
    mixed_filter_fn,
    gee_project: str = "tamiypf",
) -> None:
    """
    Registra en Colab el callback que se dispara al pulsar "🔍 Abrir La Lupa".

    Debe llamarse UNA VEZ antes de mostrar el mapa.
    Solo funciona en Google Colab.

    Uso
    ---
    ```python
    register_lupa_callback(
        master_csv=df_maestro,
        tifs_dir=PATHS["local_infer"],
        output_base_dir=PATHS["evidence_dir"],
        lee_filter_fn=lee_filter,
        mixed_filter_fn=apply_mixed_filter,
    )
    mapa = build_heatmap(df_maestro, output_path=..., enable_lupa_button=True)
    mapa
    ```
    """
    try:
        from google.colab import output as colab_output
    except ImportError:
        print("⚠️  register_lupa_callback solo funciona en Google Colab.")
        print("    Para uso local, llamar extract_evidence() directamente.")
        return

    def _lupa_handler(lon: float, lat: float) -> None:
        print(f"\n🔍 La Lupa activada para ({lat:.3f}, {lon:.3f})...")
        output_dir = output_base_dir / f"Expediente_Lon{lon:.3f}_Lat{lat:.3f}"
        extract_evidence(
            lon_target=lon,
            lat_target=lat,
            master_csv=master_csv,
            tifs_dir=tifs_dir,
            output_dir=output_dir,
            lee_filter_fn=lee_filter_fn,
            mixed_filter_fn=mixed_filter_fn,
            gee_project=gee_project,
        )

    colab_output.register_callback("lupa_kernel_callback", _lupa_handler)
    print("✅ Callback 'La Lupa' registrado. Podés hacer clic en los marcadores del mapa.")

# ---------------------------------------------------------------------------
# Extracción de evidencia (La Lupa)
# ---------------------------------------------------------------------------

def extract_evidence(
    lon_target: float,
    lat_target: float,
    master_csv: Path | pd.DataFrame,
    tifs_dir: Path,
    output_dir: Path,
    lee_filter_fn,
    mixed_filter_fn,
    gee_project: str = "tamiypf",
    panoramic_buffer_m: int | None = None,
    panoramic_px: int | None = None,
    smooth_radius_m: int | None = None,
) -> None:
    """
    Genera el expediente visual completo para una coordenada de interés.

    Por cada fecha con detección positiva produce 3 archivos:
        1. `_1_DetalleOriginal.jpg`         — imagen SAR normalizada.
        2. `_2_MascaraMLP.jpg`              — máscara binaria Otsu-Bradley.
        3. `_3_ContextoPanoramico_15km.jpg` — contexto de 15×15 km de GEE.

    Puede llamarse directamente desde el notebook O dispararse desde el mapa
    vía register_lupa_callback().
    """
    panoramic_buffer_m = panoramic_buffer_m or PREPROCESSING["panoramic_buffer_m"]
    panoramic_px       = panoramic_px       or PREPROCESSING["panoramic_px"]
    smooth_radius_m    = smooth_radius_m    or PREPROCESSING["panoramic_smooth_radius_m"]

    if isinstance(master_csv, Path):
        df = pd.read_csv(master_csv)
    else:
        df = master_csv.copy()

    df_target = df[
        (df["Lon"].round(3) == round(lon_target, 3)) &
        (df["Lat"].round(3) == round(lat_target, 3)) &
        (df["Estado_Descarga"] == "OK") &
        (df["Prediccion_IA"]   == "Slick Petroleo")
    ]

    print(f"🚨 Se encontraron {len(df_target)} detecciones positivas en esta coordenada.\n")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        ee.Initialize(project=gee_project)
    except Exception:
        pass

    for _, row in df_target.iterrows():
        fecha     = row["Fecha"]
        confianza = int(row["Confianza_IA"])
        prefijo   = f"SAR_{fecha}_Conf{confianza}%"
        print(f"📅 Procesando evento del {fecha} (Confianza de la IA: {confianza}%)...")

        # === Imágenes locales ===
        ruta_tif = tifs_dir / str(row["Ruta_Relativa"]) / row["Archivo"]
        if ruta_tif.exists():
            try:
                with rasterio.open(ruta_tif) as src:
                    matrix = src.read(1)

                from src.preprocessing import normalize_sar
                norm = normalize_sar(matrix)

                # 1. Original normalizada
                img_original = norm.astype(np.uint8)
                Image.fromarray(img_original).save(
                    output_dir / f"{prefijo}_1_DetalleOriginal.jpg"
                )

                # 2. Máscara
                img_filtrada_float = lee_filter_fn(norm.astype(np.float32))
                img_filtrada_int   = img_filtrada_float.astype(np.int32)
                img_mascara = mixed_filter_fn(img_filtrada_int, filter_fn=None)
                Image.fromarray(img_mascara).save(
                    output_dir / f"{prefijo}_2_MascaraMLP.jpg"
                )

                print("   ✔️ Imágenes de detalle guardadas.")
            except Exception as e:
                print(f"   ❌ Error generando imágenes locales: {e}")
        else:
            print("   ⚠️ No se encontró el TIF original. ¿Descomprimiste el ZIP?")

        # === Panorámica de contexto (15 km) ===
        try:
            punto     = ee.Geometry.Point([lon_target, lat_target])
            aoi_ctx   = punto.buffer(panoramic_buffer_m).bounds()
            coleccion = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(aoi_ctx)
                .filterDate(ee.Date(fecha), ee.Date(fecha).advance(1, "day"))
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            )
            if coleccion.size().getInfo() > 0:
                panorama = (
                    coleccion.select("VV")
                    .mosaic()
                    .focal_median(radius=smooth_radius_m, units="meters")
                )
                url = panorama.getThumbURL({
                    "dimensions": panoramic_px,
                    "region":     aoi_ctx,
                    "min": -25, "max": -5,
                    "format": "jpg",
                })
                resp = requests.get(url)
                if resp.status_code == 200:
                    (output_dir / f"{prefijo}_3_ContextoPanoramico_15km.jpg").write_bytes(
                        resp.content
                    )
                    print("   ✔️ Panorámica de 15x15km descargada.")
                else:
                    print("   ❌ Error descargando panorámica de Google.")
            else:
                print("   ⚠️ No se encontraron imágenes en GEE para esa fecha.")
        except Exception as e:
            print(f"   ❌ Error generando panorámica: {e}")

    print(f"\n✅ ¡Expediente Completado! Todo el material está en:\n{output_dir}")
