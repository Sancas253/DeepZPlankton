# ==============================================================
# ENTRENAMIENTO ResNet50 LEGACY — TF/Keras — Google Colab GPU
# K-Fold Cross-Validation
# ==============================================================
# Réplica fiel del mejor ResNet50 de la exploración original
# (inteligencia_computacional/GuiaClustering/AugmentResnetTrain.py),
# para medir cuánto mejoró la versión del pipeline (resnet50_mlp).
#
# Igual que el original:
#   - Keras ResNet50 (pesos "imagenet"), backbone congelado
#   - preprocess_input de ResNet (RGB→BGR y resta de la media, estilo Caffe)
#   - GAP → Dense(512, ReLU) → Dropout(0.5) → Dense(N, softmax)
#   - Adam con LR por defecto, 20 épocas fijas, sin callbacks
#   - class weights "balanced"
#   - SIN augmentación (el nombre del original engaña: no tenía)
#   - se queda con el modelo de la última época
#
# Único cambio: el esquema de evaluación es el K-Fold común del
# pipeline (mismos folds que el resto), para que el cv_results.json
# sea comparable con comparar_matrices.py.
#
# El preprocess_input va dentro del grafo: el SavedModel/ONNX recibe la
# imagen cruda [0, 255] NHWC, que es lo que el V3 entrega con
# preprocessing "efficientnet" (no tiene un preprocesado tipo Caffe).
#
# Salida: fold_k/SavedModel + fold_k/meta.json + cv_results.json
# ==============================================================
# !pip install scikit-learn -q
# ==============================================================
import os
import json
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Lambda
from tensorflow.keras.models import Model
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             confusion_matrix, classification_report)

# ==============================================================
# CONFIGURACIÓN (la del original)
# ==============================================================
ruta       = "/content/dataset_local"
OUTPUT_DIR = "/content/resnet50_legacy"
image_size = (224, 224)
batch_size = 32
seed       = 42
EPOCHS     = 20
DROPOUT    = 0.5
N_SPLITS   = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)
tf.random.set_seed(seed)
print("🚀 GPU detectada:", tf.config.list_physical_devices('GPU'))

# ==============================================================
# RECOLECTAR CLASES Y ARCHIVOS
# ==============================================================
data_dir     = pathlib.Path(ruta)
class_names  = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
num_classes  = len(class_names)
class_to_idx = {cls: i for i, cls in enumerate(class_names)}
print(f"📊 Clases detectadas ({num_classes}):", class_names)

valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
all_files, all_labels = [], []
for cls in class_names:
    for f in sorted((data_dir / cls).iterdir()):
        if f.suffix.lower() in valid_exts:
            all_files.append(str(f))
            all_labels.append(class_to_idx[cls])

all_files  = np.array(all_files)
all_labels = np.array(all_labels, dtype=np.int32)
print(f"📊 Total imágenes: {len(all_files)}")

# ==============================================================
# PIPELINE DE DATOS — sin augmentación, como el original
# (image_dataset_from_directory: resize bilineal sin conservar aspecto)
# ==============================================================
AUTOTUNE = tf.data.AUTOTUNE

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size)
    img = tf.cast(img, tf.float32)
    return img, label

def make_dataset(file_paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if training:
        ds = ds.shuffle(len(file_paths), seed=seed)
    return (ds
            .map(load_image, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE))

# preprocess_input de keras.applications.resnet50 ("caffe"), con ops de TF
# para que quede dentro del grafo exportado.
CAFFE_MEAN_BGR = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)

def preprocess_resnet(x):
    return x[..., ::-1] - CAFFE_MEAN_BGR

# ==============================================================
# K-FOLD LOOP (mismos folds que el resto del pipeline)
# ==============================================================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
fold_test_indices = [test_idx for _, test_idx in skf.split(all_files, all_labels)]

fold_results = []
oof_true, oof_pred = [], []

for fold_idx in range(N_SPLITS):
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx + 1}/{N_SPLITS}")
    print(f"{'='*60}")

    fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_output_dir, exist_ok=True)

    test_fold   = fold_idx
    val_fold    = (fold_idx + 1) % N_SPLITS
    train_folds = [i for i in range(N_SPLITS) if i != test_fold and i != val_fold]

    test_indices  = fold_test_indices[test_fold]
    val_indices   = fold_test_indices[val_fold]
    train_indices = np.concatenate([fold_test_indices[i] for i in train_folds])
    print(f"  Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")

    train_ds = make_dataset(all_files[train_indices], all_labels[train_indices], training=True)
    val_ds   = make_dataset(all_files[val_indices],   all_labels[val_indices])
    test_ds  = make_dataset(all_files[test_indices],  all_labels[test_indices])

    # Class weights sobre el train del fold (el original: sobre su split de train)
    cw = compute_class_weight("balanced", classes=np.arange(num_classes),
                              y=all_labels[train_indices])
    class_weights_dict = {i: float(w) for i, w in enumerate(cw)}

    # Modelo fresco por fold
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed + fold_idx)
    print("🔹 Cargando ResNet50 (Keras, ImageNet)...")
    base_model = ResNet50(weights="imagenet", include_top=False,
                          input_shape=(*image_size, 3))
    base_model.trainable = False

    entrada = Input(shape=(*image_size, 3), name="input")
    x      = Lambda(preprocess_resnet, name="resnet_preprocess")(entrada)
    x      = base_model(x, training=False)
    x      = GlobalAveragePooling2D()(x)
    x      = Dense(512, activation="relu")(x)
    x      = Dropout(DROPOUT)(x)
    output = Dense(num_classes, activation="softmax")(x)
    model  = Model(inputs=entrada, outputs=output)

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    print("🔒 Entrenando cabeza (ResNet50 congelado, sin augmentación, 20 épocas fijas)...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
              class_weight=class_weights_dict)

    # Métricas (modelo de la última época, como el original)
    _, last_val_acc = model.evaluate(val_ds, verbose=0)
    y_true = all_labels[test_indices]
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = y_prob.argmax(axis=1)

    test_acc      = float(accuracy_score(y_true, y_pred))
    test_f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    test_bal_acc  = float(balanced_accuracy_score(y_true, y_pred))
    test_cm       = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()
    test_report   = classification_report(y_true, y_pred, labels=list(range(num_classes)),
                                          target_names=class_names, output_dict=True,
                                          zero_division=0)
    print(f"🧪 Test fold {fold_idx}: acc={test_acc:.4f} | "
          f"f1_macro={test_f1_macro:.4f} | bal_acc={test_bal_acc:.4f}")

    fold_results.append({"fold": fold_idx, "val_acc": float(last_val_acc),
                         "test_acc": test_acc, "test_f1_macro": test_f1_macro,
                         "test_balanced_acc": test_bal_acc})
    oof_true.extend(y_true.tolist())
    oof_pred.extend(y_pred.tolist())
    # Predicciones por imagen del test (-> oof_predicciones.csv) y split.json
    import predicciones
    _rel = lambda i: os.path.relpath(all_files[i], ruta).replace("\\", "/")
    predicciones.guardar_probabilidades(
        y_prob, [_rel(i) for i in test_indices], y_true, class_names, fold_output_dir,
        predicciones.armar_split(_rel, train_indices, val_indices, test_indices))

    # Guardar SavedModel + meta.json
    model.export(fold_output_dir)
    print(f"✅ Modelo guardado: {fold_output_dir}")

    meta = {
        "framework":    "tf",
        "arch":         "tf_savedmodel",
        "image_size":   list(image_size),
        "class_names":  class_names,
        # Entrada cruda [0,255] NHWC: el preprocesado Caffe está en el grafo.
        "preprocessing": "efficientnet",
        "fold":         fold_idx,
        "n_splits":     N_SPLITS,
        "val_acc":      float(last_val_acc),
        "test_acc":     test_acc,
        "test_f1_macro":     test_f1_macro,
        "test_balanced_acc": test_bal_acc,
        "confusion_matrix":      test_cm,
        "classification_report": test_report,
        "notes": ("ResNet50 legacy (Keras, réplica de AugmentResnetTrain.py): ImageNet "
                  f"congelado, GAP→512-ReLU-Dropout({DROPOUT})→N, sin augmentación, "
                  f"{EPOCHS} épocas, class weights"),
    }
    with open(os.path.join(fold_output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# ==============================================================
# RESUMEN CROSS-VALIDATION
# ==============================================================
val_accs  = [r["val_acc"]           for r in fold_results]
test_accs = [r["test_acc"]          for r in fold_results]
test_f1s  = [r["test_f1_macro"]     for r in fold_results]
test_bals = [r["test_balanced_acc"] for r in fold_results]

cv_summary = {
    "n_splits":      N_SPLITS,
    "folds":         fold_results,
    "mean_val_acc":  float(np.mean(val_accs)),
    "std_val_acc":   float(np.std(val_accs)),
    "mean_test_acc": float(np.mean(test_accs)),
    "std_test_acc":  float(np.std(test_accs)),
    "mean_test_f1_macro":     float(np.mean(test_f1s)),
    "std_test_f1_macro":      float(np.std(test_f1s)),
    "mean_test_balanced_acc": float(np.mean(test_bals)),
    "std_test_balanced_acc":  float(np.std(test_bals)),
    "pooled_oof": {
        "n_total":    len(oof_true),
        "accuracy":   float(accuracy_score(oof_true, oof_pred)),
        "f1_macro":   float(f1_score(oof_true, oof_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(oof_true, oof_pred)),
        "confusion_matrix": confusion_matrix(oof_true, oof_pred,
                                             labels=list(range(num_classes))).tolist(),
        "classification_report": classification_report(oof_true, oof_pred,
                                                        labels=list(range(num_classes)),
                                                        target_names=class_names,
                                                        output_dict=True, zero_division=0),
    },
}

cv_path = os.path.join(OUTPUT_DIR, "cv_results.json")
with open(cv_path, "w", encoding="utf-8") as f:
    json.dump(cv_summary, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"✅ CV completo — mean_test_acc: {cv_summary['mean_test_acc']:.4f} ± {cv_summary['std_test_acc']:.4f}")
print(f"          — mean_f1_macro:     {cv_summary['mean_test_f1_macro']:.4f} ± {cv_summary['std_test_f1_macro']:.4f}")
print(f"          — mean_balanced_acc: {cv_summary['mean_test_balanced_acc']:.4f} ± {cv_summary['std_test_balanced_acc']:.4f}")
print(f"📊 OOF agrupado ({cv_summary['pooled_oof']['n_total']} imgs) — "
      f"f1_macro: {cv_summary['pooled_oof']['f1_macro']:.4f} | "
      f"balanced_acc: {cv_summary['pooled_oof']['balanced_acc']:.4f}")
print(f"✅ Resultados guardados en: {cv_path}")
print("✅ Entrenamiento ResNet50 legacy completado.")
