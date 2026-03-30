# data/

Esta carpeta contiene los datos intermedios generados por el pipeline.

## Contenido

```
data/
├── features_dataset.csv     ← generado por entrenamiento_mlp.ipynb (celda 1a)
└── kaggle_dataset/          ← generado por descarga_kaggle.ipynb (no en git)
    ├── Oil/
    └── No_Oil/
```

### `features_dataset.csv`
Vector de 16 características extraídas de cada imagen del dataset de Kaggle.
Es el input directo del entrenamiento del MLP. Se puede versionar en git
porque es liviano (texto plano).

### `kaggle_dataset/`
Imágenes JPG del dataset público de Kaggle. Se descargan con `descarga_kaggle.ipynb`.
No se incluyen en el repositorio (ver `.gitignore`).

## Datos SAR de producción

Las imágenes Sentinel-1 descargadas durante el pipeline de adquisición
(tiles GeoTIFF del Golfo San Matías) se guardan localmente en Colab o en
el path configurado en `src/config.py`. No se versionan en git por su tamaño.
