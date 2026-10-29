# =============================================================================
# Borrado de la barra de escala en imágenes de zooplancton (Google Colab)
# Copia de Training set III/eliminar_barra_colab.py (lectura/escritura con
# soporte de rutas con acentos). La usa deepz_pipeline.py como módulo.
# =============================================================================
# El software de captura estampa la escala (línea + "1 mm" / "0.5 mm" / "0.3 mm")
# siempre con la misma geometría al pixel, sin escalarla con la imagen. Por eso:
#   1. Se arma una plantilla exacta de cada variante del sello como la mediana
#      pixel a pixel de todas sus apariciones en las imágenes.
#   2. En cada imagen se ubica el sello con template matching.
#   3. Se borran exactamente los píxeles de tinta del sello, reconstruyendo cada
#      uno con material del lado que le corresponde (bicho o fondo) mediante
#      cv2.inpaint restringido a ese lado.
# Las imágenes sin sello (score de matching < SCORE_MIN) se copian sin cambios.
# Las carpetas originales nunca se modifican: la salida va a <carpeta>_barra_borrada.
# =============================================================================

import glob
import os

import cv2
import numpy as np

# ------------------------------- CONFIGURACIÓN -------------------------------
RAIZ = "/content/drive/MyDrive/Training set III/PROYECTO AUTOMATIC ZOO"

# Carpeta con muchos sellos, usada siempre para construir las plantillas.
CARPETA_REFERENCIA = "ANNELIDA_POLYCHAETA"

# Carpetas a procesar. None = todas las carpetas dentro de RAIZ
# (salvo las que ya son salidas). O una lista: ["BRYOZOA_CIFONAUTA", "CHAETOGNATA"]
CARPETAS = None

SUFIJO_SALIDA = "_barra_borrada"

# Umbral de confianza del template matching. En ANNELIDA_POLYCHAETA las imágenes
# sin sello dieron <= 0.61 y todas las que lo tienen >= 0.78.
SCORE_MIN = 0.7

# Una variante del sello necesita al menos esta fracción de los ejemplos de la
# variante más frecuente para ser plantilla (ver construir_plantillas).
FRACCION_MIN_PLANTILLA = 0.05
# -----------------------------------------------------------------------------

EXTENSIONES = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")
MARGEN = 3
ALTO_ZONA_TEXTO = 14


# Rutas a ignorar (las llena deepz_pipeline.py con las imágenes gigantes, p. ej.
# escaneos completos colados en una carpeta de clase).
EXCLUIDAS = set()


def listar_imagenes(carpeta):
    archivos = []
    for ext in EXTENSIONES:
        archivos.extend(glob.glob(os.path.join(carpeta, ext)))
        archivos.extend(glob.glob(os.path.join(carpeta, ext.upper())))
    return sorted(set(archivos) - EXCLUIDAS)


def leer_imagen(ruta):
    """cv2.imread no abre rutas con acentos en Windows (p. ej. CTENOPHORA_estómago)."""
    try:
        datos = np.fromfile(ruta, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(datos, cv2.IMREAD_COLOR) if datos.size else None


def guardar_imagen(ruta, imagen):
    ok, buf = cv2.imencode(os.path.splitext(ruta)[1], imagen)
    if not ok:
        raise IOError(f"No se pudo codificar {ruta}")
    buf.tofile(ruta)


def valor_fondo(gris):
    return float(np.percentile(gris, 90))


def buscar_linea_aislada(gris, banda=0.25):
    """Línea del sello: componente de negro casi puro, bien rectangular, arriba."""
    alto = gris.shape[0]
    zona = (gris[: max(1, int(alto * banda))] <= 60).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(zona, connectivity=8)
    mejor = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w >= 10 and w / max(h, 1) >= 4 and area / (w * h) > 0.6:
            if mejor is None or w > mejor[2]:
                mejor = (int(x), int(y), int(w), int(h))
    return mejor


def construir_plantillas(carpetas, min_ejemplos=3):
    """Una plantilla por tamaño de línea: mediana de la opacidad de la tinta."""
    recortes = {}
    for carpeta in carpetas:
        for ruta in listar_imagenes(carpeta):
            img = leer_imagen(ruta)
            if img is None:
                continue
            gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            linea = buscar_linea_aislada(gris.astype(np.uint8))
            if linea is None:
                continue
            x, y, w, h = linea
            fondo = valor_fondo(gris)
            alto_t = MARGEN + h + ALTO_ZONA_TEXTO
            ancho_t = w + 2 * MARGEN
            padded = cv2.copyMakeBorder(gris, alto_t, alto_t, ancho_t, ancho_t,
                                        cv2.BORDER_CONSTANT, value=fondo)
            y0 = y - MARGEN + alto_t
            x0 = x - MARGEN + ancho_t
            recorte = padded[y0:y0 + alto_t, x0:x0 + ancho_t]
            alpha = np.clip((fondo - recorte) / max(fondo, 1.0), 0.0, 1.0)
            recortes.setdefault((w, h), []).append(alpha)

    # Al recorrer muchas carpetas, partes oscuras y alargadas de los bichos
    # coinciden en tamaño por casualidad y forman grupos de unos pocos ejemplos
    # (en el dataset completo: 59 grupos de 3-12 contra 1396-5862 de los sellos
    # reales). Usadas como plantilla, borran trozos del bicho donde no hay sello.
    mas_comun = max((len(l) for l in recortes.values()), default=0)
    minimo = max(min_ejemplos, FRACCION_MIN_PLANTILLA * mas_comun)

    plantillas = []
    for (w, h), lista in recortes.items():
        if len(lista) < minimo:
            continue
        alpha = np.median(np.stack(lista), axis=0).astype(np.float32)
        plantillas.append({"linea": (w, h), "alpha": alpha, "n": len(lista)})
    plantillas.sort(key=lambda p: -p["linea"][0])
    return plantillas


def ubicar_sello(gris, plantillas, banda=0.3):
    alto = gris.shape[0]
    fondo = valor_fondo(gris)
    zona_base = gris[: max(1, int(alto * banda))].astype(np.float32)

    mejor = None
    for p in plantillas:
        th, tw = p["alpha"].shape
        esperado = (fondo * (1.0 - p["alpha"])).astype(np.float32)
        zona = cv2.copyMakeBorder(zona_base, th, th, tw, tw, cv2.BORDER_CONSTANT, value=fondo)
        res = cv2.matchTemplate(zona, esperado, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if mejor is None or score > mejor["score"]:
            mejor = {"score": float(score), "plantilla": p, "x": loc[0] - tw, "y": loc[1] - th}
    return mejor


def clasificar_bicho_o_fondo(bicho_conocido, fondo_conocido):
    """Cada pixel toma la etiqueta del pixel conocido (bicho/fondo) más cercano."""
    dist_bicho = cv2.distanceTransform(cv2.bitwise_not(bicho_conocido), cv2.DIST_L2, 5)
    dist_fondo = cv2.distanceTransform(cv2.bitwise_not(fondo_conocido), cv2.DIST_L2, 5)
    return dist_bicho <= dist_fondo


def rellenar_segun_forma(imagen, objetivo, bicho_conocido, fondo_conocido, radio_inpaint=5):
    """
    Rellena `objetivo` con material del lado que le corresponde a cada pixel:
    inpaint con solo píxeles de bicho como fuente donde "iría el bicho", y con
    solo fondo donde "iría el fondo". Así el borde bicho/fondo no se licúa.
    """
    hay_bicho = np.any(bicho_conocido > 0)
    hay_fondo = np.any(fondo_conocido > 0)
    if not (hay_bicho and hay_fondo):
        return cv2.inpaint(imagen, objetivo, radio_inpaint, cv2.INPAINT_TELEA)

    objetivo_bool = objetivo > 0
    es_bicho = clasificar_bicho_o_fondo(bicho_conocido, fondo_conocido)
    resultado = imagen.copy()

    for lado, fuente_excluida in ((es_bicho, fondo_conocido), (~es_bicho, bicho_conocido)):
        m = objetivo_bool & lado
        if m.any():
            hueco = cv2.bitwise_or(objetivo, fuente_excluida)
            relleno = cv2.inpaint(imagen, hueco, radio_inpaint, cv2.INPAINT_TELEA)
            resultado[m] = relleno[m]
    return resultado


def eliminar_barra(imagen, plantillas, umbral_tinta=0.12, dilatacion=1, offset_umbral=35):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    alto, ancho = gris.shape

    ubicacion = ubicar_sello(gris, plantillas) if plantillas else None
    if ubicacion is None or ubicacion["score"] < SCORE_MIN:
        return imagen, None

    alpha = ubicacion["plantilla"]["alpha"]
    th, tw = alpha.shape
    x0, y0 = ubicacion["x"], ubicacion["y"]
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(ancho, x0 + tw), min(alto, y0 + th)

    objetivo = np.zeros((alto, ancho), dtype=np.uint8)
    if ix1 > ix0 and iy1 > iy0:
        tinta = (alpha[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] > umbral_tinta)
        objetivo[iy0:iy1, ix0:ix1] = tinta.astype(np.uint8) * 255

    fondo = valor_fondo(gris)
    _, oscuro = cv2.threshold(gris, max(fondo - offset_umbral, 0), 255, cv2.THRESH_BINARY_INV)
    no_objetivo = cv2.bitwise_not(objetivo)
    bicho_conocido = cv2.bitwise_and(oscuro, no_objetivo)
    fondo_conocido = cv2.bitwise_and(cv2.bitwise_not(oscuro), no_objetivo)

    if dilatacion > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilatacion + 1,) * 2)
        objetivo = cv2.dilate(objetivo, k)

    return rellenar_segun_forma(imagen, objetivo, bicho_conocido, fondo_conocido), ubicacion["score"]


def procesar_carpeta(carpeta_entrada, plantillas):
    carpeta_salida = carpeta_entrada + SUFIJO_SALIDA
    os.makedirs(carpeta_salida, exist_ok=True)

    con_sello, sin_sello, errores = 0, 0, 0
    for ruta in listar_imagenes(carpeta_entrada):
        imagen = leer_imagen(ruta)
        if imagen is None:
            errores += 1
            continue
        limpia, score = eliminar_barra(imagen, plantillas)
        guardar_imagen(os.path.join(carpeta_salida, os.path.basename(ruta)), limpia)
        if score is None:
            sin_sello += 1
        else:
            con_sello += 1

    print(f"{os.path.basename(carpeta_entrada)}: {con_sello} con sello borrado, "
          f"{sin_sello} sin sello (copiadas igual), {errores} ilegibles -> {carpeta_salida}")


if __name__ == "__main__":
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except ImportError:
        pass

    if CARPETAS is None:
        carpetas = sorted(
            os.path.join(RAIZ, d) for d in os.listdir(RAIZ)
            if os.path.isdir(os.path.join(RAIZ, d)) and not d.endswith(SUFIJO_SALIDA)
        )
    else:
        carpetas = [os.path.join(RAIZ, d) for d in CARPETAS]

    referencia = os.path.join(RAIZ, CARPETA_REFERENCIA)
    plantillas = construir_plantillas(sorted(set([referencia] + carpetas)))
    for p in plantillas:
        print(f"Plantilla línea {p['linea'][0]}x{p['linea'][1]} px: {p['n']} ejemplos")

    for carpeta in carpetas:
        procesar_carpeta(carpeta, plantillas)
