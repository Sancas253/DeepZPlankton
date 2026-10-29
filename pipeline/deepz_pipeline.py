"""
deepz_pipeline.py — de imágenes con barra de escala a modelo ONNX para DeepZPlankton V3.

Etapas (se pueden correr todas o algunas con --etapas):
  barra     Borra la barra de escala de cada imagen (eliminar_barra.py) y arma
            un dataset limpio tipo ImageFolder (una carpeta por clase, PNG).
  entrenar  Corre el entrenamiento K-Fold elegido (entrenamiento/train_*.py,
            los mismos scripts de Modelos.ipynb) sobre el dataset limpio.
            Todos los modelos usan la augmentación predeterminada de
            aumentacion.py (se apaga con --sin-aumentacion).
  matriz    Dibuja la matriz de confusión OOF (todas las imágenes, cada una
            predicha por el fold que no la vio) desde cv_results.json.
  analizar  Por cada fold, sobre SUS imágenes de test (analizar_modelo.py
            --solo-test): Grad-CAM de los errores y UMAP de las features, en
            <entrenamiento>/analisis/fold_k. Cada entrenamiento deja además
            oof_predicciones.csv (una fila por imagen, predicha por el fold que
            no la vio) para comparar_corridas.py (McNemar, errores compartidos).
            Con --comparar-con CARPETA compara además contra todas las corridas
            anteriores de esa carpeta (-> <entrenamiento>/comparacion_anteriores).
  exportar  Exporta los folds a fold_N.onnx + meta.json (exportar_onnx.py),
            copia la matriz y cv_results.json, deja un .zip listo para
            "Agregar modelo desde ZIP" y, si existe, lo instala directo en
            dist/DeepZPlanktonV3/model/.

Modelos disponibles (--modelo):
  swin_tiny               Swin-Tiny fine-tuning (PyTorch)            224x224
  convnext_tiny           ConvNeXt-Tiny fine-tuning (PyTorch/timm)   224x224
  convnext_tiny_plus      ConvNeXt-Tiny 22k + fine-tuning completo   288x288
                          (AdamW, layer decay, coseno, EMA)
  convnext_dinov2_fusion  ConvNeXt-Tiny + DINOv2 congelados + MLP     224x224
  dinov2_mlp              DINOv2-base congelado + MLP (PyTorch)       224x224
  dinov2_ft_convnext_tiny DINOv2 con fine-tuning (4 capas + CLS) →   224x224
                          etapas 2-3 de ConvNeXt-Tiny (receta plus)
  convnext_dinov2_fusion_plus ConvNeXt-Tiny (288) ‖ DINOv2 (224) en   288x288
                          paralelo, ambos con fine-tuning
  dinov2_ft               DINOv2-base (ViT-B/14) fine-tuning completo 224x224
                          con la receta de convnext_tiny_plus (timm)
  resnet50_mlp            ResNet50 congelado + MLP (PyTorch)          224x224
  efficientnetb3          EfficientNetB3 fine-tuning (TF, via tf2onnx) 300x300
  resnet50_legacy         ResNet50 congelado + MLP (TF/Keras) réplica 224x224
                          del original de inteligencia_computacional
                          (AugmentResnetTrain.py): sin augmentación, para
                          medir cuánto aportó resnet50_mlp
  (convnext_tiny es la versión PyTorch: la de Keras no se puede pasar a ONNX.)

Entradas: --entrada (imágenes con barra) y --dataset (dataset ya limpio) aceptan
una carpeta o un .zip. Después de borrar las barras queda también
<trabajo>/dataset_limpio.zip para reusarlo con --dataset y no repetir esa etapa.
Si el dataset limpio ya existe, la etapa barra se saltea sola.

Ejemplos:
  # Todo el flujo (Colab con GPU):
  python deepz_pipeline.py --modelo swin_tiny --entrada "/content/drive/MyDrive/Training set III/PROYECTO AUTOMATIC ZOO" --trabajo /content/deepz

  # Solo exportar e instalar en el V3 (PC local) un entrenamiento bajado de Colab:
  python deepz_pipeline.py --modelo swin_tiny --etapas matriz,exportar --entrenado swin_tiny_finetuned_entrenado.zip

  # Reusar un dataset ya limpio (zip guardado de una corrida anterior):
  python deepz_pipeline.py --modelo convnext_tiny --dataset dataset_limpio.zip --trabajo /content/deepz

  # Entrenar sin algunas clases (no toca el dataset limpio original):
  python deepz_pipeline.py --modelo convnext_tiny --dataset dataset_limpio.zip --excluir "FISH_LARVAE,Trash_Hayde_Todo"

  # Probar el flujo rápido con menos folds/épocas:
  python deepz_pipeline.py --modelo dinov2_mlp --entrada ... --set N_SPLITS=2 --set EPOCHS=2
"""
import argparse
import datetime
import glob
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

# Carpeta de modelos del V3 (junto al .exe: ver gui/runtime_paths.MODEL_BASE).
DESTINO_V3 = os.path.join(os.path.dirname(AQUI), "dist", "DeepZPlanktonV3", "model")

MODELOS = {
    # nombre: (script, variable con la ruta del dataset, nombre de salida)
    "swin_tiny":              ("train_swin_tiny.py",              "RUTA_DATOS", "swin_tiny_finetuned"),
    "convnext_tiny":          ("train_convnext_tiny.py",          "RUTA_DATOS", "convnext_tiny_pt"),
    "convnext_tiny_plus":     ("train_convnext_tiny_plus.py",     "RUTA_DATOS", "convnext_tiny_plus"),
    "convnext_dinov2_fusion": ("train_convnext_dinov2_fusion.py", "RUTA_DATOS", "convnext_dinov2_fusion"),
    "dinov2_mlp":             ("train_dinov2_mlp.py",             "RUTA_DATOS", "dinov2_mlp"),
    "dinov2_ft_convnext_tiny": ("train_dinov2_ft_convnext_tiny.py", "RUTA_DATOS", "dinov2_ft_convnext_tiny"),
    "convnext_dinov2_fusion_plus": ("train_convnext_dinov2_fusion_plus.py", "RUTA_DATOS", "convnext_dinov2_fusion_plus"),
    "dinov2_ft":              ("train_dinov2_ft.py",              "RUTA_DATOS", "dinov2_ft"),
    "resnet50_mlp":           ("train_resnet50.py",               "RUTA_DATOS", "resnet50_mlp"),
    "efficientnetb3":         ("train_efficientnetb3.py",         "ruta",       "efficientnetb3_finetuned"),
    "resnet50_legacy":        ("train_resnet50_legacy.py",        "ruta",       "resnet50_legacy"),
}
# Réplicas de modelos viejos que entrenaban sin augmentación: se respeta
# aunque no se pase --sin-aumentacion.
SIN_AUMENTACION = {"resnet50_legacy"}
ETAPAS = ("barra", "entrenar", "matriz", "analizar", "exportar")
EXTENSIONES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
# Un recorte de organismo tiene ~0.02 MP (mediana) y en Training set III el más
# grande tiene 4.3 MP. Algo de decenas de MP es un escaneo completo colado en una
# carpeta de clase: no sirve para entrenar y hace lenta cada época.
MAX_MEGAPIXELES = 20
MARCA_SALIDA_BARRA = "_barra_borrada"


def titulo(texto):
    print(f"\n{'=' * 64}\n{texto}\n{'=' * 64}", flush=True)


# ------------------------------------------------------------------- zips ----

def extraer_zip(ruta_zip, destino):
    """Extrae un zip corrigiendo los nombres con acentos de los zips de Windows.

    El "Comprimir en carpeta ZIP" del Explorador guarda los nombres en la
    página de códigos OEM (cp850 en español) sin marcar UTF-8, y zipfile los
    lee como cp437: CTENOPHORA_estómago terminaría como CTENOPHORA_est¾mago.
    """
    with zipfile.ZipFile(ruta_zip) as zf:
        for info in zf.infolist():
            if not info.flag_bits & 0x800:
                try:
                    info.filename = info.filename.encode("cp437").decode("cp850")
                except UnicodeError:
                    pass
            zf.extract(info, destino)


def raiz_de_clases(carpeta):
    """Baja por carpetas contenedoras únicas (zip con 'Dataset/CLASE/...')."""
    while True:
        items = [i for i in os.listdir(carpeta) if not i.startswith(("__MACOSX", "."))]
        subdirs = [i for i in items if os.path.isdir(os.path.join(carpeta, i))]
        if len(items) == 1 and len(subdirs) == 1:
            hijo = os.path.join(carpeta, subdirs[0])
            # Si adentro hay imágenes sueltas, el único subdir ya es una clase.
            if any(os.path.isfile(os.path.join(hijo, i)) for i in os.listdir(hijo)):
                return carpeta
            carpeta = hijo
        else:
            return carpeta


def carpeta_o_zip(ruta, destino_extraccion):
    """Devuelve la carpeta con las clases: la misma ruta, o el zip extraído."""
    ruta = os.path.abspath(ruta)
    if os.path.isdir(ruta):
        return raiz_de_clases(ruta)
    if not zipfile.is_zipfile(ruta):
        raise SystemExit(f"{ruta} no es una carpeta ni un .zip")
    if os.path.isdir(destino_extraccion) and os.listdir(destino_extraccion):
        print(f"  {os.path.basename(ruta)} ya extraído en {destino_extraccion}")
    else:
        print(f"  Extrayendo {ruta} ...", flush=True)
        extraer_zip(ruta, destino_extraccion)
    return raiz_de_clases(destino_extraccion)


def comprimir_dataset(carpeta, ruta_zip):
    # Sin compresión: son PNG, deflate no gana casi nada y tarda mucho más.
    with zipfile.ZipFile(ruta_zip, "w", zipfile.ZIP_STORED) as zf:
        for raiz, _, archivos in os.walk(carpeta):
            for a in sorted(archivos):
                completo = os.path.join(raiz, a)
                zf.write(completo, os.path.relpath(completo, carpeta))
    print(f"  📦 Dataset limpio: {ruta_zip} "
          f"({os.path.getsize(ruta_zip) / 1024 / 1024:.0f} MB)")


# ------------------------------------------------------- imágenes gigantes ----

def buscar_gigantes(carpeta, max_mp):
    """[(ruta, MP, ancho, alto)] de las imágenes de más de max_mp megapíxeles.

    Lee solo el encabezado (PIL no decodifica la imagen en Image.open), así
    que recorrer decenas de miles de archivos tarda poco.
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # que no avise justo de las que buscamos
    gigantes = []
    for raiz, _, archivos in os.walk(carpeta):
        for a in sorted(archivos):
            if not a.lower().endswith(EXTENSIONES):
                continue
            ruta = os.path.join(raiz, a)
            try:
                with Image.open(ruta) as im:
                    ancho, alto = im.size
            except Exception:
                continue  # ilegible: lo cuenta quien la lea
            if ancho * alto > max_mp * 1e6:
                gigantes.append((ruta, ancho * alto / 1e6, ancho, alto))
    return gigantes


def reportar_gigantes(gigantes, base, reporte, donde):
    if not gigantes:
        return
    print(f"⚠️  {len(gigantes)} imagen(es) de más de {MAX_MEGAPIXELES} MP ({donde}), "
          f"quedan fuera del dataset:")
    with open(reporte, "a", encoding="utf-8") as f:
        for ruta, mp, ancho, alto in gigantes:
            linea = f"{os.path.relpath(ruta, base)}  {ancho}x{alto}  {mp:.0f} MP  ({donde})"
            print("   ", linea)
            f.write(linea + "\n")
    print(f"   Lista en {reporte}")


def sacar_gigantes_del_dataset(dataset, trabajo, reporte):
    """Saca del dataset limpio las imágenes gigantes antes de entrenar.

    Solo mueve archivos de la copia propia en --trabajo (p. ej. un
    dataset_limpio.zip extraído); una carpeta del usuario no se toca.
    """
    gigantes = buscar_gigantes(dataset, MAX_MEGAPIXELES)
    if not gigantes:
        return
    reportar_gigantes(gigantes, dataset, reporte, "dataset limpio")
    if os.path.commonpath([os.path.abspath(dataset), trabajo]) != trabajo:
        raise SystemExit(f"Sacá esas imágenes de {dataset} (no se borran solas de una "
                         "carpeta tuya) o subí MAX_MEGAPIXELES en deepz_pipeline.py")
    for ruta, *_ in gigantes:
        destino = os.path.join(trabajo, "descartadas", os.path.relpath(ruta, dataset))
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        shutil.move(ruta, destino)


def excluir_clases(dataset, excluir, trabajo):
    """Arma <trabajo>/dataset_filtrado sin las clases de `excluir`.

    El dataset original no se toca: las imágenes se enlazan (hardlink) o, si
    no se puede, se copian. Se rehace en cada corrida.
    """
    clases = sorted(d for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, d)))
    desconocidas = set(excluir) - set(clases)
    if desconocidas:
        raise SystemExit(f"--excluir: clases que no están en el dataset: {sorted(desconocidas)}\n"
                         f"Clases disponibles: {clases}")
    filtrado = os.path.join(trabajo, "dataset_filtrado")
    if os.path.abspath(filtrado) == os.path.abspath(dataset):
        raise SystemExit("--excluir: el dataset ya es dataset_filtrado; pasá el original")
    shutil.rmtree(filtrado, ignore_errors=True)
    for clase in clases:
        if clase in excluir:
            print(f"  {clase}: excluida")
            continue
        origen = os.path.join(dataset, clase)
        for raiz, _, archivos in os.walk(origen):
            destino_dir = os.path.join(filtrado, clase, os.path.relpath(raiz, origen))
            os.makedirs(destino_dir, exist_ok=True)
            for nombre in archivos:
                src, dst = os.path.join(raiz, nombre), os.path.join(destino_dir, nombre)
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
    print(f"Dataset filtrado: {len(clases) - len(excluir)}/{len(clases)} clases -> {filtrado}")
    return filtrado


# ------------------------------------------------------------------ barra ----

def cargar_mapeo(ruta):
    """JSON {carpeta_original: clase_destino | null}. null = descartar la carpeta."""
    if not ruta:
        return {}
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def etapa_barra(entrada, dataset, mapeo, reporte):
    import eliminar_barra as eb

    print("Buscando imágenes gigantes...", flush=True)
    gigantes = buscar_gigantes(entrada, MAX_MEGAPIXELES)
    reportar_gigantes(gigantes, entrada, reporte, "entrada")
    eb.EXCLUIDAS = {ruta for ruta, *_ in gigantes}

    carpetas = sorted(
        d for d in os.listdir(entrada)
        if os.path.isdir(os.path.join(entrada, d)) and MARCA_SALIDA_BARRA not in d
    )
    if not carpetas:
        raise SystemExit(f"No hay carpetas de clase dentro de {entrada}")

    rutas = [os.path.join(entrada, d) for d in carpetas]
    print("Construyendo plantillas del sello...", flush=True)
    plantillas = eb.construir_plantillas(rutas)
    if not plantillas:
        print("⚠️  No se encontró ningún sello: las imágenes se copian sin cambios.")
    for p in plantillas:
        print(f"  Plantilla línea {p['linea'][0]}x{p['linea'][1]} px: {p['n']} ejemplos")

    total_con, total_sin, total_err = 0, 0, 0
    for carpeta in carpetas:
        clase = mapeo.get(carpeta, carpeta)
        if clase is None:
            print(f"  {carpeta}: descartada por el mapeo")
            continue
        destino = os.path.join(dataset, clase)
        os.makedirs(destino, exist_ok=True)
        con, sin, err = 0, 0, 0
        for ruta in eb.listar_imagenes(os.path.join(entrada, carpeta)):
            imagen = eb.leer_imagen(ruta)
            if imagen is None:
                err += 1
                continue
            limpia, score = eb.eliminar_barra(imagen, plantillas)
            # Siempre PNG: tf.io.decode_image (EfficientNet) no lee TIFF y así
            # el dataset queda en el mismo formato que los segmentos del V3.
            base = os.path.splitext(os.path.basename(ruta))[0]
            salida = os.path.join(destino, base + ".png")
            n = 1
            while os.path.exists(salida):
                salida = os.path.join(destino, f"{base}_{n}.png")
                n += 1
            eb.guardar_imagen(salida, limpia)
            if score is None:
                sin += 1
            else:
                con += 1
        flecha = f" -> {clase}" if clase != carpeta else ""
        print(f"  {carpeta}{flecha}: {con} con sello borrado, {sin} sin sello, {err} ilegibles",
              flush=True)
        total_con, total_sin, total_err = total_con + con, total_sin + sin, total_err + err
    print(f"Total: {total_con} con sello borrado, {total_sin} sin sello, "
          f"{total_err} ilegibles -> {dataset}")


# --------------------------------------------------------------- entrenar ----

def parchear_script(src, asignaciones):
    """Reemplaza la línea `NOMBRE = ...` de nivel superior de cada asignación."""
    for nombre, valor in asignaciones.items():
        patron = re.compile(rf"^{re.escape(nombre)}\s*=.*$", re.MULTILINE)
        if not patron.search(src):
            raise SystemExit(f"El script de entrenamiento no define {nombre} = ...")
        src = patron.sub(lambda _m: f"{nombre} = {valor}", src, count=1)
    return src


def etapa_entrenar(modelo, dataset, salida_entreno, ajustes, aumentar=True):
    import aumentacion
    if modelo in SIN_AUMENTACION and aumentar:
        print(f"{modelo} replica un entrenamiento sin augmentación: se desactiva.")
        aumentar = False
    aumentacion.ACTIVA = aumentar
    aumentacion.guardar_config(salida_entreno)
    print(f"Augmentación: {'activada' if aumentar else 'DESACTIVADA'} "
          f"(ver {os.path.join(salida_entreno, 'aumentacion.json')})")
    script, var_datos, _ = MODELOS[modelo]
    ruta_script = os.path.join(AQUI, "entrenamiento", script)
    with open(ruta_script, encoding="utf-8") as f:
        src = f.read()

    asignaciones = {var_datos: repr(dataset), "OUTPUT_DIR": repr(salida_entreno)}
    asignaciones.update(ajustes)
    src = parchear_script(src, asignaciones)
    if os.name == "nt":
        # En Windows los workers del DataLoader re-importan __main__ y no
        # encuentran las clases definidas dentro del script ejecutado con exec.
        src = re.sub(r"num_workers\s*=\s*\d+", "num_workers=0", src)
        src = re.sub(r"^NUM_WORKERS\s*=.*$", "NUM_WORKERS = 0", src, flags=re.MULTILINE)

    print(f"Script: {script}\nDataset: {dataset}\nSalida: {salida_entreno}", flush=True)
    exec(compile(src, ruta_script, "exec"), {"__name__": "__main__", "__file__": ruta_script})
    import predicciones
    oof = predicciones.unir_oof(salida_entreno)
    if oof:
        print(f"  📄 Predicciones por imagen: {oof}")


# --------------------------------------------------------------- analizar ----

def buscar_corridas(raiz, excluir):
    """oof_predicciones.csv de corridas en raiz (<corrida>/<modelo>/ o <modelo>/), sin la actual."""
    excluir = os.path.normcase(os.path.abspath(excluir))
    encontradas = []
    for patron in ("*/*/oof_predicciones.csv", "*/oof_predicciones.csv"):
        for ruta in sorted(glob.glob(os.path.join(raiz, patron))):
            carpeta = os.path.dirname(ruta)
            if os.path.normcase(os.path.abspath(carpeta)) == excluir:
                continue
            padre = os.path.basename(os.path.dirname(carpeta))
            nombre = padre if patron.startswith("*/*") else os.path.basename(carpeta)
            encontradas.append((nombre, ruta))
    return encontradas


def comparar_con_anteriores(salida_entreno, raiz, dataset=None, min_comunes=0.95):
    """McNemar y errores compartidos de esta corrida contra todas las anteriores compatibles."""
    import comparar_corridas
    import predicciones
    actual = predicciones.leer_oof(salida_entreno)
    nombre_actual = os.path.basename(os.path.dirname(os.path.abspath(salida_entreno)))
    args = [f"{nombre_actual} (nueva)={os.path.join(salida_entreno, 'oof_predicciones.csv')}"]
    usados = {nombre_actual}
    for nombre, ruta in buscar_corridas(raiz, salida_entreno):
        otra = predicciones.leer_oof(ruta)
        comunes = set(actual) & set(otra)
        if len(comunes) < min_comunes * len(actual):
            print(f"  ↷ {nombre}: otro dataset ({len(comunes)}/{len(actual)} imágenes en común), se saltea")
            continue
        if any(actual[r]["real"] != otra[r]["real"] for r in comunes):
            print(f"  ↷ {nombre}: otras etiquetas (otro mapeo de clases), se saltea")
            continue
        while nombre in usados:
            nombre += "'"
        usados.add(nombre)
        args.append(f"{nombre}={ruta}")
    if len(args) < 2:
        print(f"No hay corridas anteriores comparables en {raiz} (con oof_predicciones.csv). "
              "Para las viejas: reconstruir_predicciones.py")
        return None
    salida = os.path.join(salida_entreno, "comparacion_anteriores")
    print(f"Comparando contra {len(args) - 1} corridas anteriores -> {salida}")
    extra = ["--dataset", dataset] if dataset else []
    comparar_corridas.main(args + ["--salida", salida] + extra)
    return salida


def etapa_analizar(salida_entreno, dataset, max_cam=40):
    """Grad-CAM + UMAP de cada fold sobre las imágenes de test que ese fold no vio."""
    import analizar_modelo
    folds = sorted((d for d in os.listdir(salida_entreno) if re.fullmatch(r"fold_\d+", d)),
                   key=lambda d: int(d.split("_")[1]))
    for fold in folds:
        carpeta = os.path.join(salida_entreno, fold)
        if not os.path.exists(os.path.join(carpeta, "model.pt")):
            print(f"  {fold}: sin model.pt (modelo TF): no hay Grad-CAM")
            continue
        if not os.path.exists(os.path.join(carpeta, "split.json")):
            print(f"  {fold}: sin split.json (entrenamiento viejo): usar reconstruir_predicciones.py")
            continue
        print(f"\n--- {fold} ---", flush=True)
        try:
            analizar_modelo.main(["--modelo", carpeta, "--imagenes", dataset, "--solo-test",
                                  "--salida", os.path.join(salida_entreno, "analisis", fold),
                                  "--max-cam", str(max_cam)])
        except (Exception, SystemExit) as e:  # el análisis nunca debe frenar el export
            print(f"⚠️  Falló el análisis de {fold}: {type(e).__name__}: {e}")


# ----------------------------------------------------------------- matriz ----

def dibujar_matriz(cm, class_names, titulo_fig, ruta_png, normalizar):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    if normalizar:
        filas = cm.sum(axis=1, keepdims=True)
        datos = np.divide(cm, filas, out=np.zeros(cm.shape, dtype=float), where=filas != 0)
        vmax = 1.0
    else:
        datos = cm
        vmax = max(int(cm.max()), 1)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(14, n * 0.45), max(12, n * 0.38)))
    im = ax.imshow(datos, cmap="Blues", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tam = max(5, 9 - n // 10)
    for i in range(n):
        for j in range(n):
            v = datos[i, j]
            if v == 0:
                continue
            txt = f"{v:.2f}" if normalizar else f"{int(v)}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=tam,
                    color="white" if v > vmax * 0.5 else "black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicho", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)
    ax.set_title(titulo_fig, fontsize=14, pad=14)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  🖼️  {ruta_png}")


def etapa_matriz(salida_entreno, modelo):
    with open(os.path.join(salida_entreno, "cv_results.json"), encoding="utf-8") as f:
        cv = json.load(f)
    with open(os.path.join(salida_entreno, "fold_0", "meta.json"), encoding="utf-8") as f:
        class_names = json.load(f)["class_names"]
    oof = cv["pooled_oof"]
    base = (f"{modelo} — OOF {oof['n_total']} imgs — acc {oof['accuracy']:.3f} | "
            f"F1 macro {oof['f1_macro']:.3f} | bal_acc {oof['balanced_acc']:.3f}")
    try:
        dibujar_matriz(oof["confusion_matrix"], class_names,
                       f"Matriz de confusión (recall por clase)\n{base}",
                       os.path.join(salida_entreno, "matriz_confusion.png"), True)
        dibujar_matriz(oof["confusion_matrix"], class_names,
                       f"Matriz de confusión (conteos)\n{base}",
                       os.path.join(salida_entreno, "matriz_confusion_conteos.png"), False)
    except ImportError:
        print("⚠️  Falta matplotlib (pip install matplotlib): la matriz queda solo en "
              "cv_results.json -> pooled_oof.confusion_matrix")


# --------------------------------------------------------------- exportar ----

ARCHIVOS_EXTRA = ("matriz_confusion.png", "matriz_confusion_conteos.png", "cv_results.json")


def etapa_exportar(salida_entreno, nombre_modelo, carpeta_resultado, destino_v3):
    from exportar_onnx import export_folds

    salida_onnx = os.path.join(carpeta_resultado, nombre_modelo)
    if os.path.exists(salida_onnx):
        raise SystemExit(f"Ya existe {salida_onnx}; borrala o usá otro --nombre")
    try:
        export_folds(salida_entreno, salida_onnx)
    except BaseException:
        shutil.rmtree(salida_onnx, ignore_errors=True)
        raise
    for extra in ARCHIVOS_EXTRA:
        origen = os.path.join(salida_entreno, extra)
        if os.path.exists(origen):
            shutil.copy2(origen, salida_onnx)

    # ZIP con los archivos sueltos en la raíz: así lo espera
    # "Agregar modelo desde ZIP" del V3 (model_manager._agregar_desde_zip).
    ruta_zip = shutil.make_archive(salida_onnx, "zip", salida_onnx)
    print(f"  📦 {ruta_zip}")

    if destino_v3:
        instalado = os.path.join(destino_v3, nombre_modelo)
        if os.path.exists(instalado):
            print(f"  ⚠️  Ya hay un modelo {instalado}; no se pisa.")
        else:
            shutil.copytree(salida_onnx, instalado)
            print(f"  ✅ Instalado en DeepZPlankton V3: {instalado}")
    return salida_onnx, ruta_zip


# ------------------------------------------------------------------- main ----

def descomprimir_entrenado(ruta_zip, destino):
    """Descomprime un zip de entrenamiento y devuelve la carpeta que tiene los fold_k/."""
    with zipfile.ZipFile(ruta_zip) as zf:
        zf.extractall(destino)
    for raiz, dirs, _ in os.walk(destino):
        if any(d.startswith("fold_") for d in dirs):
            return raiz
    raise SystemExit(f"{ruta_zip} no tiene carpetas fold_k/")


def parsear_ajustes(pares):
    ajustes = {}
    for par in pares or []:
        nombre, _, valor = par.partition("=")
        if not nombre or not valor or not re.fullmatch(r"[A-Za-z_]\w*", nombre.strip()):
            raise SystemExit(f"--set espera NOMBRE=valor, recibido: {par}")
        ajustes[nombre.strip()] = valor.strip()
    return ajustes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", required=True, choices=sorted(MODELOS))
    ap.add_argument("--entrada", help="Carpeta o .zip con una subcarpeta por clase, imágenes CON barra")
    ap.add_argument("--dataset", help="Carpeta o .zip de un dataset ya limpio (saltea la etapa barra)")
    ap.add_argument("--trabajo", default=os.path.join(os.getcwd(), "deepz_trabajo"),
                    help="Carpeta de trabajo (dataset limpio, entrenamiento, resultado)")
    ap.add_argument("--resultados",
                    help="Dónde escribir entrenamiento, matriz y ONNX a medida que se generan "
                         "(p. ej. Drive en Colab). Por defecto, --trabajo")
    ap.add_argument("--etapas", default=",".join(ETAPAS),
                    help=f"Subconjunto de {','.join(ETAPAS)} separado por comas")
    ap.add_argument("--mapeo", help="JSON {carpeta_original: clase_destino|null}")
    ap.add_argument("--excluir", metavar="CLASE1,CLASE2",
                    help="Clases del dataset limpio que no se usan para entrenar ni analizar "
                         "(separadas por comas, nombre exacto de la carpeta)")
    ap.add_argument("--entrenado", help="ZIP o carpeta de un entrenamiento ya hecho (fold_k/ + cv_results.json)")
    ap.add_argument("--nombre", help="Nombre de la carpeta del modelo en el V3")
    ap.add_argument("--destino-v3", default=DESTINO_V3 if os.path.isdir(DESTINO_V3) else "",
                    help="Carpeta model/ del V3 donde instalarlo ('' = no instalar)")
    ap.add_argument("--comparar-con", metavar="CARPETA",
                    help="En la etapa analizar: McNemar y errores compartidos contra todas las "
                         "corridas de CARPETA que tengan oof_predicciones.csv")
    ap.add_argument("--cam", type=int, default=40,
                    help="Grad-CAM por fold en la etapa analizar (primero los errores)")
    ap.add_argument("--sin-aumentacion", action="store_true",
                    help="Entrenar sin la augmentación predeterminada de aumentacion.py")
    ap.add_argument("--set", action="append", metavar="NOMBRE=valor",
                    help="Pisa una constante del script de entrenamiento (p. ej. N_SPLITS=3)")
    args = ap.parse_args()

    etapas = [e.strip() for e in args.etapas.split(",") if e.strip()]
    desconocidas = set(etapas) - set(ETAPAS)
    if desconocidas:
        raise SystemExit(f"Etapas desconocidas: {desconocidas}")

    _, _, nombre_salida = MODELOS[args.modelo]
    trabajo = os.path.abspath(args.trabajo)
    dataset = os.path.join(trabajo, "dataset_limpio")
    # Las imágenes van siempre a --trabajo (disco rápido); lo que hay que
    # conservar, a --resultados. En Colab eso es Drive: cada fold queda a salvo
    # apenas termina aunque el entorno se corte después.
    resultados = os.path.abspath(args.resultados) if args.resultados else trabajo
    reporte_gigantes = os.path.join(resultados, "imagenes_descartadas.txt")
    salida_entreno = os.path.join(resultados, nombre_salida)
    resultado = os.path.join(resultados, "resultado")
    fecha = datetime.date.today().strftime("%Y%m%d")
    nombre_modelo = args.nombre or f"{nombre_salida}_{fecha}_ensemble"
    os.makedirs(trabajo, exist_ok=True)
    os.makedirs(resultados, exist_ok=True)

    if args.dataset:
        dataset = carpeta_o_zip(args.dataset, dataset)
        print(f"Dataset limpio: {dataset}")

    if "barra" in etapas:
        titulo("1/5 · Borrado de la barra de escala")
        if os.path.isdir(dataset) and os.listdir(dataset):
            print(f"El dataset limpio ya existe ({dataset}): se reutiliza y no se "
                  "vuelven a borrar las barras. Para rehacerlo, borrá esa carpeta.")
        else:
            if not args.entrada:
                raise SystemExit("La etapa barra necesita --entrada (o --dataset ya limpio)")
            entrada = carpeta_o_zip(args.entrada, os.path.join(trabajo, "entrada_con_barra"))
            etapa_barra(entrada, dataset, cargar_mapeo(args.mapeo), reporte_gigantes)
            comprimir_dataset(dataset, os.path.join(trabajo, "dataset_limpio.zip"))

    excluir = [c.strip() for c in (args.excluir or "").split(",") if c.strip()]
    if excluir and any(e in etapas for e in ("entrenar", "analizar")):
        titulo("Exclusión de clases")
        dataset = excluir_clases(dataset, excluir, trabajo)

    if "entrenar" in etapas:
        if os.path.exists(os.path.join(salida_entreno, "cv_results.json")):
            raise SystemExit(f"{salida_entreno} ya tiene un entrenamiento completo; "
                             "movelo o salteá la etapa entrenar")
        titulo(f"2/5 · Entrenamiento {args.modelo}")
        sacar_gigantes_del_dataset(dataset, trabajo, reporte_gigantes)
        etapa_entrenar(args.modelo, dataset, salida_entreno, parsear_ajustes(args.set),
                       aumentar=not args.sin_aumentacion)
        # Entrenamiento crudo (con los .pt / SavedModel) por si el export hay
        # que rehacerlo en otra máquina (ej. efficientnetb3 en local). Con
        # --resultados la carpeta ya quedó ahí y un zip sería otra copia entera.
        if args.resultados:
            print(f"  📁 Entrenamiento crudo: {salida_entreno}")
        else:
            zip_ent = shutil.make_archive(os.path.join(trabajo, f"{nombre_salida}_entrenado"),
                                          "zip", salida_entreno)
            print(f"  📦 Entrenamiento crudo: {zip_ent}")
    elif args.entrenado:
        if os.path.isdir(args.entrenado):
            salida_entreno = os.path.abspath(args.entrenado)
        else:
            salida_entreno = descomprimir_entrenado(
                args.entrenado, tempfile.mkdtemp(prefix="deepz_entrenado_", dir=trabajo))

    if "matriz" in etapas:
        titulo("3/5 · Matriz de confusión")
        etapa_matriz(salida_entreno, args.modelo)

    if "analizar" in etapas:
        titulo("4/5 · Análisis: Grad-CAM de errores y UMAP por fold")
        if os.path.isdir(dataset) and os.listdir(dataset):
            etapa_analizar(salida_entreno, dataset, args.cam)
        else:
            print(f"No está el dataset ({dataset}): se saltea. Pasá --dataset para analizar.")
        if args.comparar_con:
            try:
                comparar_con_anteriores(salida_entreno, args.comparar_con,
                                        dataset if os.path.isdir(dataset) else None)
            except (Exception, SystemExit) as e:  # nunca frena el export
                print(f"⚠️  Falló la comparación con corridas anteriores: {type(e).__name__}: {e}")

    if "exportar" in etapas:
        titulo("5/5 · Export ONNX para DeepZPlankton V3")
        carpeta, ruta_zip = etapa_exportar(salida_entreno, nombre_modelo, resultado,
                                           args.destino_v3)
        titulo("Listo")
        print(f"Modelo ONNX: {carpeta}\nZIP para 'Agregar modelo': {ruta_zip}")


if __name__ == "__main__":
    main()
