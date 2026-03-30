"""
preprocessing.py
----------------
Funciones de filtrado y binarización de imágenes SAR.

Pipeline:
    1. lee_filter        — supresión de ruido Speckle
    2. otsu_threshold    — umbral global óptimo (Otsu 1979)
    3. bradley_filter    — umbral local adaptativo (Bradley & Roth 2007)
    4. apply_mixed_filter — combina Otsu y Bradley según el brillo medio de la escena
                           + postprocesamiento morfológico

Referencias:
    Mahdikhani & Hassannejad Bibalan (2025) — MJEE, vol. 19, no. 2.
    OpenCV Morphological Transformations — docs.opencv.org
"""

import cv2 as cv
import numpy as np

# Valores de píxel para la imagen binaria de salida
WHITE = 255
BLACK = 0


# ---------------------------------------------------------------------------
# 1. Filtro de Lee (reducción de Speckle)
# ---------------------------------------------------------------------------

def lee_filter(img: np.ndarray, size: int = 5) -> np.ndarray:
    """
    Aplica el filtro de Lee para atenuar el ruido Speckle en imágenes SAR.

    El filtro calcula, para cada píxel, un peso basado en la varianza local
    respecto de la varianza global de ruido. Preserva bordes estructurales.

    Parámetros
    ----------
    img  : np.ndarray  Imagen de entrada (float32 o convertible).
    size : int         Tamaño del kernel cuadrado (debe ser impar, default 5).

    Retorna
    -------
    np.ndarray  Imagen filtrada, misma forma que la entrada.
    """
    img = img.astype(np.float32)

    mean    = cv.blur(img,    (size, size))
    mean_sq = cv.blur(img**2, (size, size))

    local_var  = mean_sq - mean**2
    noise_var  = (np.mean(local_var))

    # Peso de Lee: 0 en zonas homogéneas, 1 en bordes
    weight = local_var / (local_var + noise_var + 1e-10)

    return mean + weight * (img - mean)


# ---------------------------------------------------------------------------
# 2. Umbral de Otsu (implementación propia)
# ---------------------------------------------------------------------------

def otsu_threshold(img: np.ndarray) -> int:
    """
    Calcula el umbral óptimo de Otsu a partir del histograma de la imagen.

    Parámetros
    ----------
    img : np.ndarray  Imagen en escala de grises (uint8 o convertible).

    Retorna
    -------
    int  Umbral T* que maximiza la varianza entre clases.
    """
    img_u8 = np.array(img, dtype=np.uint8)
    counts = np.bincount(img_u8.flatten(), minlength=256)
    prob   = counts / counts.sum()

    best_T, best_var = 0, -1.0
    for T in range(1, 256):
        w1 = prob[:T + 1].sum()
        w2 = prob[T + 1:].sum()
        if w1 == 0 or w2 == 0:
            continue
        levels = np.arange(len(prob))
        mu1  = (levels[:T + 1] * prob[:T + 1]).sum() / w1
        mu2  = (levels[T + 1:] * prob[T + 1:]).sum() / w2
        mu_T = (levels * prob).sum()
        var  = w1 * (mu1 - mu_T) ** 2 + w2 * (mu2 - mu_T) ** 2
        if var >= best_var:
            best_T, best_var = T, var

    return best_T


def otsu_binarize(img: np.ndarray, threshold: int) -> np.ndarray:
    """
    Binariza la imagen con el umbral de Otsu.
    Píxeles oscuros (≤ threshold) → BLACK; claros → WHITE.
    """
    #! NO MODIFICA IMG, la siguiente si
    # out = np.where(img <= threshold, BLACK, WHITE).astype(np.uint8)
    # return out

    mascara_light_pixeles = img > threshold
    mascara_oscuros_pixeles = img <= threshold
    img[mascara_light_pixeles] = WHITE
    img[mascara_oscuros_pixeles] = BLACK
    img = np.array(img,dtype = np.uint8)
    return img



# ---------------------------------------------------------------------------
# 3. Filtro de Bradley (umbral local adaptativo)
# ---------------------------------------------------------------------------

def bradley_filter(img: np.ndarray, window_ratio: int = 9, T: int = 23) -> np.ndarray:
    """
    Umbral local adaptativo de Bradley & Roth.

    Para cada píxel compara su intensidad contra la media local, escalada
    por el factor (1 - T/100). Implementación vectorizada con imagen integral.

    Parámetros
    ----------
    img          : np.ndarray  Imagen de entrada (int32 o convertible).
    window_ratio : int         Fracción del ancho usada como tamaño de ventana.
    T            : int         Porcentaje de sensibilidad (paper: 23).

    Retorna
    -------
    np.ndarray  Imagen binaria uint8.
    """
    img = np.array(img, dtype=np.int32)
    h, w = img.shape

    window = max(1, w // window_ratio)
    if window % 2 == 0:
        window += 1
    half = window // 2

    # Imagen integral con padding para evitar condiciones de borde
    integral = np.cumsum(np.cumsum(img, axis=0), axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")

    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    y1 = np.clip(Y - half, 0, h - 1)
    y2 = np.clip(Y + half, 0, h - 1)
    x1 = np.clip(X - half, 0, w - 1)
    x2 = np.clip(X + half, 0, w - 1)

    y1p, y2p = y1 + 1, y2 + 1
    x1p, x2p = x1 + 1, x2 + 1

    local_sum = (
        integral[y2p, x2p]
        - integral[y1p - 1, x2p]
        - integral[y2p, x1p - 1]
        + integral[y1p - 1, x1p - 1]
    )
    area      = (y2 - y1 + 1) * (x2 - x1 + 1)
    local_mean = local_sum / area
    threshold  = local_mean * (100 - T) / 100

    return np.where(img < threshold, BLACK, WHITE).astype(np.uint8)


# ---------------------------------------------------------------------------
# 4. Filtro mixto Otsu-Bradley + morfología
# ---------------------------------------------------------------------------

def apply_mixed_filter(
    img: np.ndarray,
    filter_fn=None,
    radius: int = 1,
) -> np.ndarray:
    """
    Combina los filtros de Otsu y Bradley con una regla de decisión basada en
    el brillo global de la escena (Ecuación 4 del paper), y aplica
    postprocesamiento morfológico para suavizar contornos y eliminar artefactos.

    Regla de coordinación:
        Si mean(img) < otsu_T  →  usar máscara de Bradley
        Si mean(img) ≥ otsu_T  →  intersección lógica (Otsu AND Bradley)

    Postprocesamiento:
        Dilatación  con disco de radio (2r+1)
        Erosión     con disco de radio (2r)

    Parámetros
    ----------
    img       : np.ndarray  Imagen SAR normalizada (int32).
    filter_fn : callable | None  Función de prefiltrado opcional
                (e.g. lee_filter). Si es None, se usa la imagen tal cual.
    radius    : int  Radio del elemento estructurante (default 1).

    Retorna
    -------
    np.ndarray  Imagen binaria uint8.
    """
    filtered = filter_fn(img) if filter_fn is not None else img.copy()

    M = (np.mean(img))
    D = otsu_threshold(img)

    A    = otsu_binarize(filtered, D)
    B = bradley_filter(filtered)

    if M < D:
        #combined = B
        filtered = B
    else:
        #combined = cv.bitwise_and(A, B) #!

        filtered = A & B 

    # Morfología: dilatar → erosionar
    disk_d = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    disk_e = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (2 * radius, 2 * radius)
    )
    
    # combined = cv.dilate(combined, disk_d)
    # combined = cv.erode(combined,  disk_e)

    # return combined.astype(np.uint8)

    filtered = cv.dilate(filtered, disk_d)
    filtered = cv.erode(filtered,  disk_e)

    return filtered.astype(np.uint8)


# ---------------------------------------------------------------------------
# Utilidad: normalización de imagen SAR (dB → uint8)
# ---------------------------------------------------------------------------

def normalize_sar(
    matrix: np.ndarray,
    db_min: float = -25.0,
    db_max: float =   0.0,
    nodata_fill: float = -25.0,
) -> np.ndarray:
    """
    Normaliza una matriz SAR de valores dB al rango [0, 255].

    Los NaN (píxeles sin datos / bordes orbitales) se reemplazan por `nodata_fill`
    antes de recortar al rango [db_min, db_max].

    Parámetros
    ----------
    matrix      : np.ndarray  Matriz SAR cruda (float, valores en dB).
    db_min      : float       Valor mínimo del rango dB (default -25).
    db_max      : float       Valor máximo del rango dB (default 0).
    nodata_fill : float       Valor de reemplazo para NaN (default -25).

    Retorna
    -------
    np.ndarray  Imagen normalizada float32 en [0, 255].
    """
    filled  = np.nan_to_num(matrix, nan=nodata_fill)
    clipped = np.clip(filled, db_min, db_max)
    return ((clipped - db_min) / (db_max - db_min) * 255.0)
