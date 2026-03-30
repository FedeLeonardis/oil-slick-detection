# 🛢️ Detección Automática de Oil Slicks en Imágenes SAR

**TAMI 2026 · En colaboración con YPF**

Pipeline de detección automática de manchas de hidrocarburos (*oil slicks*) en
imágenes de Radar de Apertura Sintética (SAR) del satélite Sentinel-1, aplicado
al Golfo de San Matías, Río Negro, Argentina.

---

## 📄 Paper

El informe técnico esta en proceso, su borrador está disponible en [`docs/informe_tami2026_borrador.pdf`](docs/informe_tami2026_borrador.pdf).

> L. Bailez, G. Bordenave Vazquez, D. Bustos, U. Cañellas, F. Leonardis Ayala,
> M. Lista, L. Martinelli, A. Porto, M. Rolando, P. Scollo.
> *Detección automática de Oil Slicks mediante imágenes SAR*, TAMI 2026.

---

## 🏗️ Estructura del repositorio

```
oil-slick-detection/
│
├── src/                        ← Módulos Python del pipeline
│   ├── config.py               ← Parámetros globales (AOI, rutas, modelo)
│   ├── acquisition.py          ← Descarga concurrente desde Google Earth Engine
│   ├── preprocessing.py        ← Filtro Lee + binarización Otsu-Bradley
│   ├── features.py             ← Extracción de 16 características
│   ├── model.py                ← Arquitectura MLP (PyTorch) e inferencia
│   └── visualization.py        ← Mapa de calor Folium + auditoría visual
│
├── notebooks/
│   ├── pipeline_principal.ipynb  ← Notebook orquestador
│   └── entrenamiento_mlp.ipynb   ← Entrenamiento y evaluación del modelo
│   └── descarga_kaggle.ipynb   ← Descarga del dataset de entrenamiento
│
├── models/                     ← Artefactos del modelo
│   ├── feature_net_model.pth   ← Pesos del OilSlickMLP
│   └── scaler.joblib           ← StandardScaler ajustado al dataset
│
├── data/
│   ├── README.md               ← Descripción del contenido de la carpeta
│
├── docs/
│   └── paper_tami2026_borrador.pdf      ← Informe técnico
│
├── requirements.txt
└── .gitignore
```

---

## 🔄 Pipeline

```
Sentinel-1 (GEE)
      │
      ▼
[Filtro ETOPO1]  →  Solo cuadrantes marinos (elev. < -5 m)
[Filtro ERA5]    →  Solo capturas con viento 3–10 m/s
      │
      ▼
Tiles 400×400 px (VV, 10 m/px)
      │
      ▼
[Filtro de Lee]             →  Reducción de Speckle
[Otsu-Bradley + Morfología] →  Binarización adaptativa
      │
      ▼
Extracción de 16 características (geometría + intensidad + textura)
      │
      ▼
OilSlickMLP (PyTorch)       →  Clasificación binaria + confianza
      │
      ▼
Mapa de calor interactivo (Folium) + Auditoría visual (La Lupa)
```

---

## ⚡ Opción A — Google Colab (recomendado para descarga de datos)

### 1. Configurar Google Cloud y GEE

Antes de correr cualquier notebook, necesitás un proyecto de Google Cloud con la API de Earth Engine habilitada:

1. Ir a [console.cloud.google.com](https://console.cloud.google.com) y crear un proyecto nuevo (o usar uno existente).
2. Buscar **"Earth Engine API"** en la biblioteca de APIs y habilitarla.
3. Registrar el proyecto en [signup.earthengine.google.com](https://signup.earthengine.google.com) — puede tardar.
4. Anotar el **Project ID** (ej. `mi-proyecto-gee`).
5. Abrir `src/config.py` y reemplazar:
   ```python
   GEE_PROJECT = "tamiypf"   # <- poner tu Project ID acá
   ```

### 2. Clonar el repositorio en Drive

En una celda de Colab:
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/FedeLeonardis/oil-slick-detection.git \
    /content/drive/MyDrive/oil-slick-detection
```

### 3. Copiar los artefactos del modelo

Subir a `Mi unidad/oil-slick-detection/models/`:
- `feature_net_model.pth`
- `scaler.joblib`

### 4. Autenticar GEE (se pide automáticamente al ejecutar la celda 0)

Al correr el notebook, se ejecuta `ee.Authenticate()`, que abre un link para
autorizar el acceso. Seguir el flujo OAuth y pegar el código que devuelve.
Esta autenticación dura la sesión de Colab; hay que repetirla cada vez que se
reinicia el runtime.

### 5. Ejecutar el notebook

Abrir `notebooks/pipeline_principal.ipynb` en Colab y ejecutar las celdas en orden.

---

## 💻 Opción B — Ejecución local (Falta testear)

### 1. Requisitos previos

- Python 3.10 o superior
- `git`
- Cuenta de Google Cloud con GEE habilitado (ver paso 1 de la sección Colab)

### 2. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/oil-slick-detection.git
cd oil-slick-detection
```

### 3. Crear y activar el entorno virtual

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Una vez activado, el prompt muestra `(.venv)` al inicio.

### 4. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ `rasterio` a veces requiere dependencias del sistema (GDAL).
> En Ubuntu/Debian: `sudo apt install gdal-bin libgdal-dev`
> En macOS con Homebrew: `brew install gdal`

### 5. Autenticar Google Earth Engine localmente

A diferencia de Colab, la autenticación local genera un token persistente:

```bash
earthengine authenticate
```

Esto abre el navegador para autorizar el acceso. El token se guarda en
`~/.config/earthengine/credentials` y no hay que repetirlo en cada ejecución.

Luego inicializar con tu Project ID:
```python
import ee
ee.Initialize(project="mi-proyecto-gee")
```

O simplemente editar `GEE_PROJECT` en `src/config.py` y el pipeline lo toma automáticamente.

### 6. Adaptar las rutas para ejecución local

En `src/config.py`, las rutas por defecto apuntan a Google Drive (`/content/drive/...`).
Para uso local, redefinirlas al inicio del notebook o modificar directamente el archivo:

```python
# src/config.py — sección PATHS, reemplazar con rutas locales
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent   # raíz del repo

PATHS = {
    "images_dir":   BASE_DIR / "data" / "imagenes_sar",
    "metadata_dir": BASE_DIR / "data" / "metadatos",
    "models_dir":   BASE_DIR / "models",
    "evidence_dir": BASE_DIR / "data" / "evidencias",
    "master_csv":   BASE_DIR / "data" / "metadatos" / "Registro_Maestro_Golfo.csv",
    "images_zip":   BASE_DIR / "data" / "imagenes_sar_local.zip",
    "local_tifs":   BASE_DIR / "data" / "tifs_temporales",
    "local_infer":  BASE_DIR / "data" / "tifs_inferencia",
}
```

### 7. Ejecutar los notebooks localmente

```bash
jupyter notebook
```

Abrir `notebooks/pipeline_principal.ipynb` desde la interfaz de Jupyter.

> El mapa Folium se guarda como HTML y se puede abrir en cualquier navegador.
> En Colab se muestra inline; localmente abrirlo con:
> ```python
> import webbrowser
> webbrowser.open("ruta/al/Mapa_Calor_Resultados.html")
> ```

---

## 📁 Estructura de carpetas en Google Drive

Al correr el pipeline en Colab, se genera automáticamente esta estructura
en `Mi unidad/TAMI_YPF_Global/` (configurable en `src/config.py`):

```
Mi unidad/
└── TAMI_YPF_Global/
    │
    ├── Modelos/                        ← subir manualmente antes de inferir
    │   ├── feature_net_model.pth
    │   └── scaler.joblib
    │
    ├── 1_Imagenes_SAR/                 ← generado por acquisition.py
    │   └── (vacío — las imágenes van al ZIP)
    │
    ├── 1_Imagenes_SAR_Local.zip        ← ZIP de todos los TIF descargados
    │                                      (generado al terminar la adquisición)
    │
    ├── 2_Metadatos/
    │   └── Registro_Maestro_Golfo.csv  ← CSV maestro con metadatos + predicciones
    │                                      (se actualiza después de cada etapa)
    │
    ├── Evidencias_Visuales/            ← generado por La Lupa
    │   └── Expediente_Lon_Lat/
    │       ├── SAR_FECHA_1_DetalleOriginal.jpg
    │       ├── SAR_FECHA_2_MascaraMLP.jpg
    │       └── SAR_FECHA_3_ContextoPanoramico_15km.jpg
    │
    └── Mapa_Calor_Resultados.html      ← mapa interactivo de detecciones
```

### Flujo de datos

```
GEE → TIFs en SSD local de Colab → ZIP en Drive
                                  → CSV maestro en Drive (checkpoint)
```

Los TIFs se descargan primero a la SSD local de Colab (rápido) y al terminar
la adquisición se comprimen en un ZIP que queda persistido en Drive. Así no
se pierden al cerrar la sesión.

---

Todos los parámetros del pipeline están centralizados en un solo archivo:

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `GEE_PROJECT` | `"tamiypf"` | **Cambiar por tu Project ID de GCP** |
| `AOI` | Golfo San Matías | Bounding box del área de interés |
| `ACQUISITION["date_start/end"]` | `2024-01-01 / 2024-12-31` | Rango temporal |
| `ACQUISITION["wind_min/max_ms"]` | `3.0 / 10.0` | Rango de viento válido (ERA5) |
| `MODEL["confidence_threshold"]` | `60.0` | Confianza mínima para reportar detección |

---

## 📊 Resultados del modelo

Evaluación sobre el dataset de Kaggle (split 75/25):

| Métrica | KNN | **MLP** |
|---------|-----|---------|
| Precisión | 84.0% | **85.0%** |
| Recall | 64.0% | **75.4%** |
| F1-Score | 73.0% | **80.0%** |
| Accuracy | 83.8% | **87.2%** |

---

## 🔮 Trabajo futuro

- Pulir la lógica de tiles para eliminar falsos positivos por bordes orbitales
- Ampliar el dataset de entrenamiento con más variabilidad oceánica
- Búsqueda exhaustiva de hiperparámetros del MLP
- Análisis de correlación entre las 16 características → selección de features
- Migrar el pipeline a una aplicación web unificada

---

## 👥 Autores

L. Bailez · G. Bordenave Vazquez · D. Bustos · U. Cañellas · F. Leonardis Ayala ·
M. Lista · L. Martinelli · A. Porto · M. Rolando · P. Scollo

*Universidad de Buenos Aires · UNMdP · UNICEN · UNSAM*

---

## 📚 Referencias principales

- Brekke & Solberg (2005). *Oil spill detection by satellite remote sensing*. RSE.
- Mahdikhani & Hassannejad Bibalan (2025). *Detection of oil slicks in SAR satellite
  images using Otsu-Bradley's thresholding method*. MJEE.
- Topouzelis et al. (2008). *Detection and discrimination between oil spills and
  look-alike phenomena through neural networks*. ISPRS.
