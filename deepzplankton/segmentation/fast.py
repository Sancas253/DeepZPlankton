# -*- coding: utf-8 -*-
"""
Segmentador final SIN visualización:
- cálculo de métricas
- generación de UN solo Excel acumulativo
- ejecución más rápida (sin imshow / waitKey)
"""
import os

import cv2
import numpy as np

from ..core.exceptions import ProcesoCancelado
from ..core.files import enlazar_o_copiar, liberar_destino
from ..core.paths import data_path
from .detection import detectar_contornos, expandir_bbox
from .metrics import MIN_CIRCULARITY, circularity, eccentricity, fill_ratio, solidity
from .results import guardar_resultados

BUFFER = 20


def Segmenter_No_View(imagenes, MIN_AREA=1000, on_progress=None, should_stop=None):

    OUTPUT_SEG = data_path("Segmentos")
    OUTPUT_CACHE = data_path("Segmentos", "_Cache")
    OUTPUT_MET = data_path("Datos")

    os.makedirs(OUTPUT_SEG, exist_ok=True)
    os.makedirs(OUTPUT_CACHE, exist_ok=True)
    os.makedirs(OUTPUT_MET, exist_ok=True)

    excel_path = os.path.join(OUTPUT_MET, "Mediciones.xlsx")
    all_results = []
    total_imagenes = len(imagenes)
    cancelado = False

    for idx, fname in enumerate(imagenes):

        if should_stop is not None and should_stop():
            print("\n⏹ Segmentación cancelada por el usuario.")
            cancelado = True
            break

        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
            if on_progress is not None:
                on_progress(idx + 1, total_imagenes)
            continue

        print(f"\n📄 Procesando: {fname}")

        base = os.path.splitext(os.path.basename(fname))[0]
        campaign_out = os.path.join(OUTPUT_SEG, base)
        cache_out = os.path.join(OUTPUT_CACHE, base)

        os.makedirs(campaign_out, exist_ok=True)
        os.makedirs(cache_out, exist_ok=True)

        img = cv2.imread(fname)

        if img is None:
            print(f"⚠ No se pudo leer {fname}")
            if on_progress is not None:
                on_progress(idx + 1, total_imagenes)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cnts = detectar_contornos(gray)

        obj_id = 0
        H, W = gray.shape

        for c in cnts:

            area = cv2.contourArea(c)
            if area < MIN_AREA:
                continue

            circ = circularity(c)
            if circ < MIN_CIRCULARITY:
                continue

            x, y, w, h = cv2.boundingRect(c)
            x1, y1, x2, y2 = expandir_bbox(x, y, w, h, BUFFER, W, H)
            crop = img[y1:y2, x1:x2]

            outname = f"{base}_{obj_id}.jpg"

            # ================= MÉTRICAS =================
            sol = solidity(c)
            exc = eccentricity(c)
            tex = np.std(crop[:, :, 0]) if crop.ndim == 3 else np.std(crop)
            fill = fill_ratio(area, w, h)

            # ================= GUARDADO =================
            save_path = os.path.join(campaign_out, outname)
            cache_path = os.path.join(cache_out, outname)

            # El recorte se comprime una sola vez; el caché es un enlace al
            # mismo contenido. Se desvincula el destino antes de escribir porque
            # re-segmentar la misma muestra reusa los nombres ({base}_{n}.jpg) y
            # con otro min_area ese nombre puede ser otro organismo: pisarlo en
            # el lugar le cambiaría los píxeles a lo que ya está asentado.
            liberar_destino(save_path)
            cv2.imwrite(save_path, crop)
            enlazar_o_copiar(save_path, cache_path)

            # ================= REGISTRO =================
            all_results.append({
                "segment": outname,
                "x": x, "y": y, "width": w, "height": h, "area": area,
                "circularity": circ, "solidity": sol, "eccentricity": exc,
                "texture_std": tex, "fill_ratio": fill,
            })

            obj_id += 1

        print(f"   → {obj_id} organismos detectados para {fname}")

        if on_progress is not None:
            on_progress(idx + 1, total_imagenes)

    guardar_resultados(excel_path, all_results)

    if cancelado:
        print(f"⚠ Segmentación incompleta: se guardó lo procesado hasta la cancelación.")
        raise ProcesoCancelado("Segmentación cancelada por el usuario.")

    print("\n✅ Segmentación completa")
    print(f"📁 Segmentos guardados en: {OUTPUT_SEG}")
    print(f"📊 Excel actualizado en: {excel_path}")
