"""
features.py
-----------
Extracción del vector de 16 características geométricas, de intensidad
y texturales a partir de la máscara binaria y la imagen SAR original.

Estas características son la entrada directa del modelo MLP.

Categorías
----------
Geometría  (8): área, perímetro, excentricidad, solidez, extensión,
                circularidad, elongación, convexidad.
Intensidad (5): media, desvío estándar, mediana, coef. variación, contraste.
Textura    (2): varianza local, entropía.
Topología  (1): número de regiones detectadas.
"""

from __future__ import annotations

import numpy as np
import cv2 as cv
from skimage.measure import regionprops, label
from skimage.filters.rank import entropy as skimage_entropy
from skimage.morphology import disk

# Valor de píxel que representa el fondo (ver preprocessing.py)
WHITE = 255
BLACK = 0

# Nombre canónico de las 16 características (orden fijo → coincide con el scaler)
FEATURE_NAMES = [
    # Geometría
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "extent",
    "circularity",
    "elongation",
    "convexity",
    # Intensidad
    "mean_intensity",
    "std_intensity",
    "median_intensity",
    "coeff_variation",
    "contrast_background",
    # Textura
    "local_variance",
    "entropy",
    # Topología
    "num_regions",
]


def extract_features(
    original: np.ndarray,
    binary_mask: np.ndarray,
    entropy_disk_radius: int = 3,
) -> dict[str, float] | None:
    """
    Extrae el vector de 16 características para una imagen SAR binarizada.

    La función trabaja sobre la **región más grande** de la máscara binaria,
    lo cual es suficiente para discriminar manchas de petróleo de look-alikes.

    Parámetros
    ----------
    original           : np.ndarray  Imagen SAR normalizada (int32 o float).
    binary_mask        : np.ndarray  Salida de `apply_mixed_filter` (uint8).
    entropy_disk_radius: int         Radio del disco para la entropía local.

    Retorna
    -------
    dict[str, float]  Diccionario con los 16 valores de características.
    None              Si no se detecta ninguna región en la máscara.
    """
    # Invertir la máscara: las manchas oscuras del SAR quedan como blanco
    # mask = binary_mask.copy()
    # mask[mask == WHITE] = 128   # temporal
    # mask[mask == BLACK] = WHITE
    # mask[mask == 128]   = BLACK
    # mask = (mask == WHITE).astype(np.uint8)

    filtered = binary_mask.copy()

    # invert mask
    mask_white = filtered == WHITE
    mask_black = filtered == BLACK
    filtered[mask_white] = BLACK
    filtered[mask_black] = WHITE

    mask = (filtered == WHITE).astype(np.uint8)


    labeled  = label(mask)
    regions  = regionprops(labeled)

    if not regions:
        return None

    # Seleccionar la región de mayor área
    region = max(regions, key=lambda r: r.area)

    area      = region.area
    perimeter = region.perimeter

    # ------------------------------------------------------------------
    # Características geométricas
    # ------------------------------------------------------------------
    eccentricity = region.eccentricity
    solidity     = region.solidity
    extent       = region.extent
    circularity  = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    major = region.major_axis_length
    minor = region.minor_axis_length
    elongation = major / minor if minor > 0 else 0.0

    convex_area = region.convex_area
    convexity   = area / convex_area if convex_area > 0 else 0.0

    # ------------------------------------------------------------------
    # Características de intensidad
    # ------------------------------------------------------------------
    coords = region.coords                           # shape (N, 2)
    pixels = original[coords[:, 0], coords[:, 1]]

    mean_intensity   = (np.mean(pixels))
    std_intensity    = (np.std(pixels))
    median_intensity = (np.median(pixels))
    coeff_variation  = std_intensity / (mean_intensity + 1e-6)

    background_pixels = original[mask == 0]
    background_mean   = (np.mean(background_pixels)) if background_pixels.size > 0 else mean_intensity
    contrast          = background_mean - mean_intensity

    # ------------------------------------------------------------------
    # Características de textura
    # ------------------------------------------------------------------
    local_variance = (np.var(pixels))

    entropy_img    = skimage_entropy(original.astype(np.uint8), disk(entropy_disk_radius))
    entropy_region = (np.mean(entropy_img[coords[:, 0], coords[:, 1]]))

    # ------------------------------------------------------------------
    # Topología
    # ------------------------------------------------------------------
    num_regions = len(regions)

    return {
        "area":               (area),
        "perimeter":          (perimeter),
        "eccentricity":       eccentricity,
        "solidity":           solidity,
        "extent":             extent,
        "circularity":        circularity,
        "elongation":         elongation,
        "convexity":          convexity,
        "mean_intensity":     mean_intensity,
        "std_intensity":      std_intensity,
        "median_intensity":   median_intensity,
        "coeff_variation":    coeff_variation,
        "contrast_background":contrast,
        "local_variance":     local_variance,
        "entropy":            entropy_region,
        "num_regions":        (num_regions),
    }


def features_to_array(features: dict[str, float]) -> np.ndarray:
    """
    Convierte el diccionario de características en un array 1-D con el
    orden canónico definido en `FEATURE_NAMES`.

    Útil para asegurarse de que el orden de columnas coincide con el scaler.

    Parámetros
    ----------
    features : dict[str, float]  Salida de `extract_features`.

    Retorna
    -------
    np.ndarray  Array float32 de forma (16,).
    """
    return np.array([features[k] for k in FEATURE_NAMES], dtype=np.float32)
