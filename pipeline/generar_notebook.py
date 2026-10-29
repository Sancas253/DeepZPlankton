"""
generar_notebook.py — arma Pipeline_DeepZ.ipynb con el pipeline embebido adentro.

El notebook lleva todos los scripts de esta carpeta (deepz_pipeline.py,
eliminar_barra.py, exportar_onnx.py, entrenamiento/*.py) comprimidos en base64
y los escribe en /content/deepz_pipeline al arrancar. Así a Colab se sube un
solo archivo y nunca puede quedar un notebook nuevo con scripts viejos en Drive.

Correrlo cada vez que se cambie cualquier script:
    python generar_notebook.py
"""
import base64
import datetime
import io
import json
import os
import zipfile

AQUI = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(AQUI, "Pipeline_DeepZ.ipynb")
ARCHIVOS = ["deepz_pipeline.py", "eliminar_barra.py", "exportar_onnx.py",
            "aumentacion.py", "mapeo_clases_ejemplo.json", "modelo_fusion_plus.py",
            "predicciones.py", "analizar_modelo.py", "comparar_corridas.py",
            "reconstruir_predicciones.py"]
ARCHIVOS += sorted(os.path.join("entrenamiento", f)
                   for f in os.listdir(os.path.join(AQUI, "entrenamiento"))
                   if f.endswith(".py"))


def empaquetar():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in ARCHIVOS:
            zf.write(os.path.join(AQUI, rel), rel.replace(os.sep, "/"))
    return base64.b64encode(buf.getvalue()).decode("ascii")


MD = """# DeepZPlankton — pipeline de entrenamiento
Imágenes **con barra de escala** → borrado de la barra → entrenamiento K-Fold → **matriz de confusión** → **ONNX listo para DeepZPlankton V3**. Todo en una sola celda: configurás, le das ▶ y te vas.

**Lo único que hay que subir es este notebook**: el pipeline viene adentro. Activá GPU en *Entorno de ejecución → Cambiar tipo de entorno*.

- **Entrada:** carpeta **o** `.zip` en Drive con una subcarpeta por clase (las que terminan en `_barra_borrada` se ignoran). Con zip la copia desde Drive es mucho más rápida.
- **Todo resultado se escribe directo en Drive, a medida que se genera.** En `RESULTADOS_DRIVE` queda `dataset_limpio.zip` (compartido entre corridas) y una carpeta por corrida, `<modelo>_<fecha_hora>/`, con:
  - `log.txt`: toda la salida, actualizada en vivo.
  - `config.json`: con qué parámetros se corrió.
  - `<modelo>/fold_k/`: pesos de cada fold apenas termina, más `cv_results.json`, `aumentacion.json` y las matrices de confusión en PNG.
  - `resultado/<modelo>_ensemble.zip`: el ONNX (→ V3: *Agregar modelo desde ZIP*), y la misma carpeta descomprimida.
- **Augmentación:** todos los modelos entrenan con la misma augmentación predeterminada (volteos, rotación ±12°, traslación, zoom, brillo/contraste, blur ocasional; fondo blanco). Se define en `aumentacion.py` y se apaga con `AUMENTACION`.
- En el disco de Colab quedan solo las imágenes (con y sin barra), que es lo que necesita velocidad.
- **Si Colab se corta a mitad del entrenamiento:** los folds terminados y el log ya están en Drive. Tildá `REUSAR_LIMPIO` y volvé a correr: arranca directo del dataset sin barras guardado (el entrenamiento empieza de cero, en una carpeta de corrida nueva).
- Colab gratis puede desconectar sesiones largas aunque esté corriendo; con Colab Pro o con el navegador abierto es más seguro."""

CELDA = r'''# @title DeepZPlankton · barra → entrenamiento → matriz → ONNX
MODELO = "swin_tiny"  # @param ["swin_tiny", "convnext_tiny", "convnext_tiny_plus", "convnext_dinov2_fusion", "dinov2_mlp", "convnext_dinov2_fusion_plus", "dinov2_ft", "resnet50_mlp", "efficientnetb3", "resnet50_legacy"]
# Carpeta o .zip con las imágenes CON barra
ENTRADA_DRIVE = "/content/drive/MyDrive/Training set III/PROYECTO AUTOMATIC ZOO"  # @param {type:"string"}
RESULTADOS_DRIVE = "/content/drive/MyDrive/deepz_resultados"  # @param {type:"string"}
# Usar RESULTADOS_DRIVE/dataset_limpio.zip de una corrida anterior (saltea el borrado de barras)
REUSAR_LIMPIO = False  # @param {type:"boolean"}
# JSON opcional en Drive {carpeta_original: clase_destino | null} para renombrar/unir/descartar clases
MAPEO_JSON = ""  # @param {type:"string"}
# Augmentación predeterminada (aumentacion.py) al entrenar; destildar para entrenar sin
AUMENTACION = True  # @param {type:"boolean"}
# Constantes del script a pisar, separadas por espacio. Ej: "N_SPLITS=3 EPOCHS=10"
AJUSTES = ""  # @param {type:"string"}
# Apagar el entorno al terminar (no gasta GPU de más; todo queda en Drive)
APAGAR_AL_TERMINAR = False  # @param {type:"boolean"}

import base64, datetime, glob, io, json, os, shutil, subprocess, sys, time, zipfile
from google.colab import drive
from IPython.display import Image, display

# -- Pipeline embebido (lo genera generar_notebook.py; no editar a mano).
PIPELINE_GENERADO = "__FECHA__"
PIPELINE_ZIP_B64 = "__B64__"
PIPELINE_DIR = "/content/deepz_pipeline"
shutil.rmtree(PIPELINE_DIR, ignore_errors=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PIPELINE_ZIP_B64))) as zf:
    zf.extractall(PIPELINE_DIR)
SCRIPT_PIPELINE = os.path.join(PIPELINE_DIR, "deepz_pipeline.py")
print(f"Pipeline embebido (generado {PIPELINE_GENERADO}) → {PIPELINE_DIR}")

drive.mount("/content/drive")
if not os.path.exists(ENTRADA_DRIVE) and not REUSAR_LIMPIO:
    raise FileNotFoundError(f"No existe ENTRADA_DRIVE: {ENTRADA_DRIVE}")
if MAPEO_JSON and not os.path.exists(MAPEO_JSON):
    raise FileNotFoundError(f"No existe MAPEO_JSON: {MAPEO_JSON}")

TRABAJO = "/content/deepz"
os.makedirs(RESULTADOS_DRIVE, exist_ok=True)
LIMPIO_DRIVE = os.path.join(RESULTADOS_DRIVE, "dataset_limpio.zip")
if REUSAR_LIMPIO and not os.path.exists(LIMPIO_DRIVE):
    raise FileNotFoundError(f"No existe {LIMPIO_DRIVE}: desmarcá REUSAR_LIMPIO")
# Carpeta de esta corrida en Drive: entrenamiento, matriz, ONNX y log van
# directo acá a medida que se generan (el disco de Colab se borra al cortarse).
CORRIDA = os.path.join(RESULTADOS_DRIVE, f"{MODELO}_{datetime.datetime.now():%Y%m%d_%H%M}")
os.makedirs(CORRIDA, exist_ok=True)
LOG = open(os.path.join(CORRIDA, "log.txt"), "a", encoding="utf-8")
with open(os.path.join(CORRIDA, "config.json"), "w", encoding="utf-8") as f:
    json.dump({"MODELO": MODELO, "ENTRADA_DRIVE": ENTRADA_DRIVE, "REUSAR_LIMPIO": REUSAR_LIMPIO,
               "MAPEO_JSON": MAPEO_JSON, "AJUSTES": AJUSTES, "AUMENTACION": AUMENTACION,
               "PIPELINE_GENERADO": PIPELINE_GENERADO}, f, ensure_ascii=False, indent=2)
print("Resultados de esta corrida:", CORRIDA)
T0 = time.time()


def mostrar(texto):
    """A pantalla y al log de Drive."""
    sys.stdout.write(texto)
    sys.stdout.flush()
    LOG.write(texto)
    LOG.flush()


def paso(texto):
    mostrar(f"\n{'#' * 70}\n# {texto}  [{(time.time() - T0) / 60:.0f} min]\n{'#' * 70}\n")


def correr(args):
    """Corre un comando mostrando la salida en vivo; si falla, frena todo."""
    import codecs
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         env={**os.environ, "PYTHONUNBUFFERED": "1"})
    # Leer por bloques (no por línea) para que las barras de tqdm, que usan
    # \r, se vean en vivo en vez de aparecer de golpe al final de cada época.
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while True:
        bloque = os.read(p.stdout.fileno(), 4096)
        if not bloque:
            break
        mostrar(dec.decode(bloque))
    if p.wait() != 0:
        mostrar(f"\n❌ Falló (código {p.returncode}). Detalle arriba y en {LOG.name}\n")
        raise RuntimeError(f"Falló (código {p.returncode}): {' '.join(args[:4])} ...")


def a_local(origen, destino):
    """Copia de Drive al disco de Colab una sola vez (zip o carpeta)."""
    if os.path.exists(destino):
        return destino
    t0 = time.time()
    if os.path.isdir(origen):
        shutil.copytree(origen, destino, ignore=shutil.ignore_patterns("*_barra_borrada*"))
    else:
        shutil.copy2(origen, destino)
    mostrar(f"Copiado {origen} en {(time.time() - t0) / 60:.1f} min\n")
    return destino


# -- Dependencias. protobuf<6: onnx>=1.20 arrastra protobuf 7, que choca con lo
#    que trae Colab (TensorFlow, grpc, google-*). Con el tope pip elige onnx 1.19.
paso("Instalando dependencias")
paquetes = ["timm", "transformers", "scikit-learn", "onnx", "onnxruntime", "onnxscript", "protobuf<6", "umap-learn"]
if MODELO in ("efficientnetb3", "resnet50_legacy"):
    paquetes.append("tf2onnx")
correr([sys.executable, "-m", "pip", "install", "-q", *paquetes])

PIPELINE = [sys.executable, SCRIPT_PIPELINE,
            "--modelo", MODELO, "--trabajo", TRABAJO, "--resultados", CORRIDA,
            "--destino-v3", ""]

# -- 1. Dataset sin barras (se guarda en Drive apenas está, por si se corta).
paso("Copiando datos al disco de Colab")
if REUSAR_LIMPIO:
    origen = ["--dataset", a_local(LIMPIO_DRIVE, "/content/dataset_limpio.zip")]
else:
    destino = "/content/entrada_con_barra" + (".zip" if ENTRADA_DRIVE.lower().endswith(".zip") else "")
    origen = ["--entrada", a_local(ENTRADA_DRIVE, destino)]
    if MAPEO_JSON:
        origen += ["--mapeo", MAPEO_JSON]

correr(PIPELINE + origen + ["--etapas", "barra"])
zip_limpio = os.path.join(TRABAJO, "dataset_limpio.zip")
if os.path.exists(zip_limpio) and not REUSAR_LIMPIO:
    shutil.copy2(zip_limpio, LIMPIO_DRIVE)
    mostrar(f"💾 Dataset sin barras guardado en Drive: {LIMPIO_DRIVE}\n")

# -- 2-5. Entrenamiento, matriz de confusión, análisis (Grad-CAM/UMAP) y export ONNX.
paso(f"Entrenamiento {MODELO} → matriz → análisis → ONNX")
ajustes = [x for a in AJUSTES.split() for x in ("--set", a)]
if not AUMENTACION:
    ajustes.append("--sin-aumentacion")
# Si falla el export, el entrenamiento crudo ya está en CORRIDA/<modelo>/ y se
# re-exporta en la PC con --etapas matriz,exportar --entrenado <esa carpeta>.
correr(PIPELINE + ["--etapas", "entrenar,matriz,analizar,exportar",
                   "--comparar-con", RESULTADOS_DRIVE] + ajustes)

# -- Resultados.
paso("Resultados")
for png in sorted(glob.glob(f"{CORRIDA}/*/matriz_confusion*.png")):
    display(Image(png))
for resumen in glob.glob(f"{CORRIDA}/*/comparacion_anteriores/resumen.md"):
    from IPython.display import Markdown
    display(Markdown(open(resumen, encoding="utf-8").read()))
    display(Image(os.path.join(os.path.dirname(resumen), "comparacion.png")))
modelos_onnx = glob.glob(f"{CORRIDA}/resultado/*.zip")
mostrar(f"\n✅ Listo en {(time.time() - T0) / 60:.0f} min. Todo en Drive: {CORRIDA}\n"
        f"   En DeepZPlankton V3 → Agregar modelo desde ZIP → {', '.join(modelos_onnx)}\n")
LOG.close()

if APAGAR_AL_TERMINAR:
    drive.flush_and_unmount()
    from google.colab import runtime
    runtime.unassign()
'''


def main():
    fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    celda = CELDA.replace("__FECHA__", fecha).replace("__B64__", empaquetar())
    nb = {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": MD.splitlines(keepends=True)},
            # cellView "form": Colab muestra solo el formulario y esconde el
            # código (incluida la línea larga del pipeline embebido).
            {"cell_type": "code", "metadata": {"cellView": "form"}, "execution_count": None,
             "outputs": [], "source": celda.splitlines(keepends=True)},
        ],
    }
    with open(SALIDA, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"{SALIDA}  ({os.path.getsize(SALIDA) / 1024:.0f} KB, {len(ARCHIVOS)} archivos embebidos)")


if __name__ == "__main__":
    main()
