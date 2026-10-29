"""Motor de inferencia ONNX y orquestación de la clasificación.

Agrupa a propósito, en un solo módulo, todo lo que comparte el caché de
sesiones ONNX en tiempo de ejecución (_MODEL_CACHE, _load_onnx_sessions,
_classify_onnx, classifier): son piezas que se llaman entre sí por nombre de
módulo y que los tests ejercitan parcheando ese estado compartido, así que
separarlas en archivos distintos rompería esa dependencia sin ganar nada real
en legibilidad. Lo que sí es independiente (nombrado de archivos, registro de
preprocesados, manifest, Excel) vive en sus propios módulos y se reexporta acá
para mantener una única puerta de entrada pública.
"""
import os

import numpy as np

from .exceptions import ProcesoCancelado
from .excel_io import append_to_excel, merge_predictions_into_excel, normalize_prediction_columns
from .files import vaciar_carpeta
from .meta import REQUIRED_FIELDS, load_meta
from .naming import con_confianza, sin_confianza
from .paths import model_key, model_path
from .preprocessing import PREPROC_LAYOUT, PREPROCESSORS

__all__ = [
    "ProcesoCancelado", "vaciar_carpeta", "load_meta", "REQUIRED_FIELDS",
    "con_confianza", "sin_confianza", "PREPROCESSORS", "PREPROC_LAYOUT",
    "append_to_excel", "normalize_prediction_columns",
    "load_images_as_batch", "evict_model_cache", "classifier",
]


# ------------------------------------------------------------
# Image loading (shared) — uses PIL, no TF dependency
# image_size: (H, W) convention matching TF's target_size
# ------------------------------------------------------------
def load_images_as_batch(file_paths, image_size):
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor

    h, w = int(image_size[0]), int(image_size[1])

    def _load_one(p):
        img = Image.open(p).convert("RGB").resize((w, h))  # PIL resize takes (W, H)
        return np.array(img, dtype=np.float32)

    # Decodificar/resizear es I/O + CPU-bound en PIL (libera el GIL durante el
    # decode), así que paralelizarlo por imagen acelera batches grandes. El
    # orden se preserva: ThreadPoolExecutor.map devuelve los resultados en el
    # mismo orden que file_paths, que es lo que zip() más abajo asume.
    with ThreadPoolExecutor(max_workers=min(8, len(file_paths))) as pool:
        imgs = list(pool.map(_load_one, file_paths))
    return np.stack(imgs, axis=0)  # shape: (B, H, W, 3)


# ------------------------------------------------------------
# Model cache — avoid reloading from disk on every call
# ------------------------------------------------------------
_MODEL_CACHE: dict = {}


def evict_model_cache(model_dir=None):
    """Descarta modelos cacheados. Sin argumento vacía todo el caché.

    Las claves se arman con la ruta del modelo, así que al reemplazar un modelo
    en el mismo destino (Agregar modelo → mismo nombre) se seguirían sirviendo
    los pesos viejos hasta reiniciar la app.
    """
    if model_dir is None:
        _MODEL_CACHE.clear()
        return
    clave = model_key(model_dir)
    for key in [k for k in _MODEL_CACHE
                if k == clave or (isinstance(k, tuple) and k[0] == clave)]:
        del _MODEL_CACHE[key]


# ------------------------------------------------------------
# ONNX inference (single model o ensemble de folds)
# ------------------------------------------------------------
def _load_onnx_sessions(model_dir, best_fold_only=False, best_fold=None):
    """Carga una o varias sesiones ONNX según la estructura del directorio.

    - fold_0.onnx, fold_1.onnx … → ensemble (un modelo por fold)
    - model.onnx                 → modelo único (comportamiento legacy)
    - best_fold_only=True        → carga solo fold_{best_fold}.onnx

    Devuelve lista de sesiones (len=1 para modelo único o mejor fold).
    """
    import onnxruntime as ort

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("🎮 Placa de video (GPU) detectada, usando aceleración CUDA.")
    else:
        providers = ["CPUExecutionProvider"]
        print("ℹ️ No se detectó placa de video (GPU), usando CPU.")

    base = model_path(model_dir)

    fold_files = sorted(
        f for f in os.listdir(base)
        if f.startswith("fold_") and f.endswith(".onnx")
    )
    if fold_files:
        if best_fold_only and best_fold is not None:
            target = f"fold_{best_fold}.onnx"
            if target in fold_files:
                print(f"🔹 Cargando mejor fold ONNX (fold_{best_fold})...")
                sessions = [ort.InferenceSession(os.path.join(base, target), providers=providers)]
                print(f"✅ Mejor fold cargado.\n")
                return sessions
            print(f"⚠️ fold_{best_fold}.onnx no encontrado, cargando ensemble completo.")
        print(f"🔹 Cargando ensemble ONNX ({len(fold_files)} folds)...")
        sessions = [
            ort.InferenceSession(os.path.join(base, f), providers=providers)
            for f in fold_files
        ]
        print(f"✅ Ensemble cargado ({len(sessions)} modelos).\n")
        return sessions

    # Fallback: model.onnx único
    # Ojo con el nombre: model_path es la función importada de paths que se usa
    # arriba para resolver `base`. Asignarle acá la ruta del .onnx la convertía
    # en local de toda la función y esa primera llamada explotaba con
    # UnboundLocalError, o sea que no cargaba ningún modelo, ni ensemble ni único.
    single_path = os.path.join(base, "model.onnx")
    print("🔹 Cargando modelo ONNX...")
    sessions = [ort.InferenceSession(single_path, providers=providers)]
    print("✅ Modelo ONNX cargado.\n")
    return sessions


def _classify_onnx(model_dir, meta, file_paths, batch_size, best_fold_only=False,
                    on_progress=None, should_stop=None):
    image_size = tuple(meta["image_size"])
    class_names = meta["class_names"]
    preprocessing = meta["preprocessing"]

    preprocess = PREPROCESSORS.get(preprocessing)
    if preprocess is None:
        raise ValueError(
            f"Preprocessing '{preprocessing}' no soportado para ONNX. "
            f"Disponibles: {list(PREPROCESSORS.keys())}"
        )

    cache_key = (model_key(model_dir), best_fold_only)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = _load_onnx_sessions(
            model_dir, best_fold_only=best_fold_only, best_fold=meta.get("best_fold")
        )
    sessions = _MODEL_CACHE[cache_key]

    # El manifest y el grafo tienen que declarar el mismo nº de clases: si no,
    # las etiquetas salen corridas y no hay ningún error visible.
    out_dim = sessions[0].get_outputs()[0].shape[-1]
    if isinstance(out_dim, int) and out_dim != len(class_names):
        raise ValueError(
            f"meta.json declara {len(class_names)} clases pero el modelo produce "
            f"{out_dim} salidas. Las etiquetas no corresponden a este modelo."
        )

    from concurrent.futures import ThreadPoolExecutor

    def _run_fold(sess_and_tensor):
        sess, tensor = sess_and_tensor
        inp = sess.get_inputs()[0].name
        logits = sess.run(None, {inp: tensor})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    pred_dict = {}
    total_batches = (len(file_paths) + batch_size - 1) // batch_size
    # Un solo pool para todos los batches: ONNX RT es thread-safe y las
    # sessions no cambian entre batches, así que crear/destruir el pool en
    # cada vuelta del loop era overhead puro.
    with ThreadPoolExecutor(max_workers=len(sessions)) as pool:
        for batch_idx, i in enumerate(range(0, len(file_paths), batch_size)):
            if should_stop is not None and should_stop():
                raise ProcesoCancelado("Clasificación cancelada por el usuario.")

            batch_paths = file_paths[i:i + batch_size]
            print(f"  Batch {batch_idx + 1}/{total_batches}...")
            batch = load_images_as_batch(batch_paths, image_size)
            tensor = preprocess(batch)

            # Folds en paralelo — cada fold en su propio thread
            all_probs = list(pool.map(_run_fold, [(s, tensor) for s in sessions]))
            probs = np.mean(all_probs, axis=0)

            y_pred = np.argmax(probs, axis=1)
            for path_img, prob_vec, pred_idx in zip(batch_paths, probs, y_pred):
                pred_dict[os.path.basename(path_img).strip()] = (
                    class_names[pred_idx], float(np.max(prob_vec))
                )

            if on_progress is not None:
                on_progress(batch_idx + 1, total_batches)
    return pred_dict


def _gather_images(ruta_test):
    """Recolecta imágenes de ruta_test, deduplicadas por nombre de archivo.

    Dos archivos con el mismo nombre en subcarpetas distintas se pisarían en
    pred_dict y en disco: se ignora el segundo y se avisa.
    """
    valid_ext = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    file_paths = []
    for root, dirs, files in os.walk(ruta_test):
        for f in files:
            if f.lower().endswith(valid_ext):
                file_paths.append(os.path.join(root, f))
    print(f"📸 Imágenes encontradas: {len(file_paths)}")

    seen_basenames: dict = {}
    deduped: list = []
    for p in file_paths:
        name = os.path.basename(p).strip()
        if name not in seen_basenames:
            seen_basenames[name] = p
            deduped.append(p)
        else:
            print(f"⚠ Nombre duplicado ignorado: '{name}' (ya registrado desde {seen_basenames[name]})")
    if len(deduped) < len(file_paths):
        print(f"⚠ {len(file_paths) - len(deduped)} archivo(s) omitidos por nombre duplicado.")
    return deduped


def _copiar_resultados(file_paths, pred_dict, ruta_salida, umbral):
    """Limpia copias viejas de imágenes que se reclasifican y copia las nuevas
    a ruta_salida/CLASE/ (o ruta_salida/_DUDOSOS/CLASE/ si conf < umbral).

    ruta_salida es EnRevision/, la carpeta que se revisa a mano: ahí el nombre
    lleva la confianza adelante para poder ordenar por duda, y si hay umbral lo
    dudoso va a su propia subcarpeta _DUDOSOS/CLASE para que se note a simple
    vista en el explorador de archivos, no solo por el prefijo del nombre.
    """
    names_to_reclassify = set(pred_dict.keys())

    # Eliminar copias viejas de imágenes que se van a reclasificar. Se compara
    # sin el prefijo: la copia vieja quedó con la confianza del modelo anterior
    # ("061_X.jpg") y la nueva lleva otra ("048_X.jpg"), así que comparar el
    # nombre tal cual dejaría el mismo segmento dos veces.
    if os.path.exists(ruta_salida):
        for clase_dir in os.scandir(ruta_salida):
            if not clase_dir.is_dir():
                continue
            if clase_dir.name == "_DUDOSOS":
                for sub_clase_dir in os.scandir(clase_dir.path):
                    if not sub_clase_dir.is_dir():
                        continue
                    for f in os.scandir(sub_clase_dir.path):
                        if sin_confianza(f.name, names_to_reclassify) in names_to_reclassify:
                            os.remove(f.path)
                continue
            for f in os.scandir(clase_dir.path):
                if sin_confianza(f.name, names_to_reclassify) in names_to_reclassify:
                    os.remove(f.path)

    for path_img in file_paths:
        nombre = os.path.basename(path_img).strip()
        if nombre in pred_dict:
            pred_label, conf = pred_dict[nombre]
            es_dudoso = umbral is not None and conf < umbral
            base_dir = os.path.join(ruta_salida, "_DUDOSOS") if es_dudoso else ruta_salida
            destino_carpeta = os.path.join(base_dir, pred_label)
            os.makedirs(destino_carpeta, exist_ok=True)
            from .files import enlazar_o_copiar
            enlazar_o_copiar(path_img,
                             os.path.join(destino_carpeta,
                                          con_confianza(nombre, conf, names_to_reclassify)))


# ------------------------------------------------------------
# Main classifier — called by la GUI
# ------------------------------------------------------------
def classifier(model_dir, ruta_test, ruta_salida, excel_path, best_fold_only=False,
                output_excel_path=None, on_progress=None, should_stop=None, umbral=None):
    batch_size = 16

    model_dir = model_path(model_dir)
    meta = load_meta(model_dir)
    framework = meta["framework"]

    if framework != "onnx":
        raise ValueError(
            f"framework debe ser 'onnx', recibido: '{framework}'.\n"
            "La app ya no ejecuta SavedModel de TensorFlow ni checkpoints .pt; "
            "convertí el modelo con average_and_export_onnx.py."
        )

    if meta["preprocessing"] not in PREPROCESSORS:
        raise ValueError(
            f"Unknown preprocessing '{meta['preprocessing']}'. "
            f"Available: {list(PREPROCESSORS.keys())}"
        )

    print("🔍 Buscando imágenes en:", ruta_test)
    file_paths = _gather_images(ruta_test)
    if not file_paths:
        print("⚠ No se encontraron imágenes.")
        return

    print("🔹 Ejecutando inferencia por batches...")
    pred_dict = _classify_onnx(model_dir, meta, file_paths, batch_size,
                               best_fold_only=best_fold_only,
                               on_progress=on_progress, should_stop=should_stop)

    _copiar_resultados(file_paths, pred_dict, ruta_salida, umbral)
    print("✅ Clasificación completada.")

    # Update Excel BEFORE clearing cache so images are safe if this fails
    print("📘 Actualizando Excel...")
    merge_predictions_into_excel(pred_dict, excel_path, output_excel_path)

    # El caché de segmentos se conserva a propósito: así "Clasificar" se puede
    # volver a correr sobre los mismos segmentos (p. ej. para probar otro
    # modelo) tantas veces como haga falta. Antes se vaciaba acá y la segunda
    # corrida no encontraba nada. Lo limpian igual quien corresponde: cada
    # segmentación nueva (WorkerThread) y cada clasificación con imágenes
    # arrastradas (ClassifierThread) lo vacían antes de llenarlo.
