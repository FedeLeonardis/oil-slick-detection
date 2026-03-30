"""
model.py
--------
Definición de la arquitectura del MLP y lógica de inferencia.

Arquitectura
------------
OilSlickMLP: Perceptrón Multicapa de 3 capas totalmente conectadas.
    Input (16) → Linear(32) → ReLU → Linear(16) → ReLU → Linear(2)

La salida es un vector de logits para dos clases:
    0 → Falsa Alarma (look-alike natural)
    1 → Slick Petroleo

Las probabilidades se obtienen con softmax. La clase predicha es el argmax.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Arquitectura
# ---------------------------------------------------------------------------

class OilSlickMLP(nn.Module):
    """
    Perceptrón Multicapa para clasificación binaria de oil slicks.

    Parámetros
    ----------
    input_dim   : int  Número de características de entrada (default 16).
    num_classes : int  Número de clases de salida (default 2).
    """

    def __init__(self, input_dim: int = 16, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Carga del modelo entrenado
# ---------------------------------------------------------------------------

def load_model(
    weights_path: str | Path,
    scaler_path:  str | Path,
    input_dim: int = 16,
    num_classes: int = 2,
    device: str | torch.device | None = None,
) -> tuple[OilSlickMLP, object, torch.device]:
    """
    Carga los pesos del modelo MLP y el scaler desde disco.

    Parámetros
    ----------
    weights_path : str | Path  Ruta al archivo .pth con los state_dict.
    scaler_path  : str | Path  Ruta al archivo .joblib con el StandardScaler.
    input_dim    : int         Dimensión de entrada del modelo.
    num_classes  : int         Número de clases.
    device       : str | None  'cuda', 'cpu', o None (autodetectar).

    Retorna
    -------
    (modelo, scaler, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = OilSlickMLP(input_dim=input_dim, num_classes=num_classes)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    scaler = joblib.load(scaler_path)

    return model, scaler, device


# ---------------------------------------------------------------------------
# Inferencia sobre un vector de características
# ---------------------------------------------------------------------------

def predict(
    features: dict[str, float] | np.ndarray | pd.DataFrame,
    model: OilSlickMLP,
    scaler,
    device: torch.device,
    feature_names: list[str] | None = None,
    label_positive: str = "Slick Petroleo",
    label_negative: str = "Falsa Alarma",
) -> tuple[str, float]:
    """
    Clasifica un vector de características como oil slick o falsa alarma.

    Parámetros
    ----------
    features       : dict | np.ndarray | pd.DataFrame
                     Vector de 16 características (una sola muestra).
    model          : OilSlickMLP  Modelo cargado en eval mode.
    scaler         : sklearn scaler  StandardScaler ajustado al dataset.
    device         : torch.device
    feature_names  : list[str] | None  Orden de columnas (solo si features es dict).
    label_positive : str  Etiqueta de la clase positiva.
    label_negative : str  Etiqueta de la clase negativa.

    Retorna
    -------
    (etiqueta, confianza_pct)  Tupla con la clase predicha y la confianza en %.
    """
    # Normalizar siempre a numpy array 2D — evita el warning de sklearn
    # "X has feature names, but StandardScaler was fitted without feature names"
    #! PROBABLEMENTE ESTE ACA EL ERROR 
    # if isinstance(features, dict):
    #     if feature_names is None:
    #         from src.features import FEATURE_NAMES
    #         feature_names = FEATURE_NAMES
    #     arr = np.array([[features[k] for k in feature_names]], dtype=np.float64)
    # elif isinstance(features, pd.DataFrame):
    #     arr = features.values
    # else:
    #     arr = np.array(features).reshape(1, -1)

    arr = pd.DataFrame([features])
    scaled  = scaler.transform(arr.values)
    tensor  = torch.FloatTensor(scaled).to(device)

    with torch.no_grad():
        logits       = model(tensor)
        probs        = torch.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    label      = label_positive if pred.item() == 1 else label_negative
    confidence_pct = (confidence.item()) * 100.0

    return label, confidence_pct


# ---------------------------------------------------------------------------
# Inferencia por lotes sobre un DataFrame del CSV maestro
# ---------------------------------------------------------------------------

def run_inference_batch(
    df_maestro: pd.DataFrame,
    model: OilSlickMLP,
    scaler,
    device: torch.device,
    tifs_dir: Path,
    extract_fn,
    max_nodata_fraction: float = 0.05,
    confidence_threshold: float = 60.0,
) -> pd.DataFrame:
    """
    Corre inferencia sobre todas las filas pendientes del CSV maestro.

    Replica exactamente el pipeline del notebook original de inferencia:
        1. normalize_sar(matriz)
        2. lee_filter(matriz_norm.astype(float32))
        3. apply_mixed_filter(img_proc, filter_fn=None)
        4. extract_features(img_proc, mask)
        5. scaler.transform + modelo MLP

    Solo procesa filas donde:
        Estado_Descarga == 'OK'  AND  Prediccion_IA == 'Pendiente'

    Parámetros
    ----------
    df_maestro          : pd.DataFrame  CSV maestro cargado.
    model               : OilSlickMLP
    scaler              : sklearn scaler
    device              : torch.device
    tifs_dir            : Path  Directorio raíz de los TIF descomprimidos.
    extract_fn          : callable  `extract_features` de features.py.
    max_nodata_fraction : float  Fracción máxima de NaN tolerada.
    confidence_threshold: float  No usado en inferencia, reservado.

    Retorna
    -------
    pd.DataFrame  DataFrame actualizado con predicciones.
    """
    import rasterio
    from src.preprocessing import normalize_sar, lee_filter, apply_mixed_filter

    pending = df_maestro[
        (df_maestro["Estado_Descarga"] == "OK") &
        (df_maestro["Prediccion_IA"]   == "Pendiente")
    ]
    print(f"🔎 Procesando {len(pending)} imágenes pendientes...")

    for idx, row in pending.iterrows():
        ruta_tif = tifs_dir / str(row["Ruta_Relativa"]) / row["Archivo"]
        if not ruta_tif.exists():
            continue

        try:
            with rasterio.open(ruta_tif) as src:
                matriz = src.read(1)

            # Filtro de bordes orbitales
            nodata_fraction = (np.isnan(matriz).sum()) / matriz.size
            if nodata_fraction > max_nodata_fraction:
                df_maestro.at[idx, "Prediccion_IA"] = "Descartado (Borde Satelite)"
                continue

            # Replicación exacta del notebook original de inferencia
            matriz_norm  = normalize_sar(matriz)
            img_filtered = lee_filter(matriz_norm.astype(np.float32))
            img_proc     = img_filtered.astype(np.int32)
            mask         = apply_mixed_filter(img_proc, filter_fn=None)
            features     = extract_fn(img_proc, mask)

            if features is None:
                df_maestro.at[idx, "Prediccion_IA"] = "Agua Limpia"
                continue

            label, conf = predict(features, model, scaler, device)

            df_maestro.at[idx, "Prediccion_IA"] = label
            df_maestro.at[idx, "Confianza_IA"]  = round(conf, 2)
            print(f"   ✔️ {row['Archivo']} -> {label} ({conf:.1f}%)")

        except Exception as exc:
            df_maestro.at[idx, "Prediccion_IA"] = "Error"
            print(f"   ✗ Error en {row['Archivo']}: {exc}")

    return df_maestro
