"""
oil-slick-detection — src package
==================================
Módulos del pipeline de detección de oil slicks en imágenes SAR Sentinel-1.

    config          — Parámetros globales (AOI, fechas, modelo, rutas)
    acquisition     — Descarga concurrente desde Google Earth Engine
    preprocessing   — Filtro Lee, Otsu, Bradley, filtro mixto, morfología
    features        — Extracción de 16 características geométricas y texturales
    model           — Arquitectura MLP (PyTorch) e inferencia
    visualization   — Mapa de calor Folium y auditoría visual (La Lupa)
"""
