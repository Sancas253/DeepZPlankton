# -*- coding: utf-8 -*-
"""
Segmentador final integrado con:
- cálculo de métricas
- generación de UN solo Excel acumulativo
- vista en vivo (imshow) para corte interactivo por tecla
"""
import os

import cv2
import numpy as np

from ..core.files import enlazar_o_copiar, liberar_destino
from ..core.paths import data_path
from .detection import detectar_contornos, expandir_bbox
from .metrics import MIN_CIRCULARITY, circularity, eccentricity, fill_ratio, solidity
from .results import guardar_resultados

BUFFER = 20
MAX_DISPLAY = 1500


def scale_for_display(img, max_size=1500):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1:
        return img, 1
    return cv2.resize(img, (int(w * scale), int(h * scale))), scale


def Segmenter_With_View(imagenes, MIN_AREA=1000):

    OUTPUT_SEG = data_path("Segmentos")
    OUTPUT_CACHE = data_path("Segmentos", "_Cache")
    OUTPUT_MET = data_path("Datos")

    os.makedirs(OUTPUT_SEG, exist_ok=True)
    os.makedirs(OUTPUT_CACHE, exist_ok=True)
    os.makedirs(OUTPUT_MET, exist_ok=True)

    excel_path = os.path.join(OUTPUT_MET, "Mediciones.xlsx")
    all_results = []

    for fname in imagenes:

        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
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
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cnts = detectar_contornos(gray)
        img_disp, disp_scale = scale_for_display(img, MAX_DISPLAY)

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

            # ================= VISUALIZACIÓN =================
            shown = img_disp.copy()
            cv2.rectangle(
                shown,
                (int(x1 * disp_scale), int(y1 * disp_scale)),
                (int(x2 * disp_scale), int(y2 * disp_scale)),
                (0, 255, 0), 2,
            )
            cv2.imshow("Imagen Completa", shown)

            crop_disp, _ = scale_for_display(crop, 400)
            cv2.imshow("Segmented Image", crop_disp)

            key = cv2.waitKey(150)
            if key == ord("q"):
                break

            outname = f"{base}_{obj_id}.jpg"

            # ================= MÉTRICAS =================
            sol = solidity(c)
            exc = eccentricity(c)
            tex = np.std(crop[:, :, 0]) if crop.ndim == 3 else np.std(crop)
            fill = fill_ratio(area, w, h)

            # ================= GUARDADO =================
            save_path = os.path.join(campaign_out, outname)
            cache_path = os.path.join(cache_out, outname)

            # Ver la nota en fast.py: una sola compresión, el caché es un
            # enlace, y nunca se escribe encima de un nombre reusado.
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

    cv2.destroyAllWindows()
    guardar_resultados(excel_path, all_results)

    print("\n✅ Segmentación completa")
    print(f"📁 Segmentos guardados en: {OUTPUT_SEG}")
    print(f"📊 Excel actualizado en: {excel_path}")
