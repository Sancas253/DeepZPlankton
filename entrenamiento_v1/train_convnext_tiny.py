# ==============================================================
# ENTRENAMIENTO ConvNeXt-Tiny — TF/Keras — Google Colab GPU
# K-Fold Cross-Validation
# ==============================================================
# Fase 1: backbone congelado, solo cabeza
# Fase 2: fine-tuning de las últimas 30 capas del backbone
# Salida: fold_k/SavedModel + fold_k/meta.json + cv_results.json
# ==============================================================
# Instalar dependencias en Colab si es necesario:
#   !pip install scikit-learn -q
# ==============================================================
import os
import json
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.applications.convnext import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             confusion_matrix, classification_report)

# ==============================================================
# CONFIGURACIÓN
# ==============================================================
RUTA_DATOS   = "/content/dataset_local"
OUTPUT_DIR   = "/content/convnext_tiny_finetuned"
IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 32
SEED         = 42
EPOCHS_HEAD     = 15
EPOCHS_FINETUNE = 20
N_SPLITS        = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)
tf.random.set_seed(SEED)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"🚀 GPU detectada: {[g.name for g in gpus]}")
else:
    print("⚠️  No GPU detectada — TF corre en CPU.")
    print("   → En Colab: Runtime → Cambiar tipo de entorno de ejecución → GPU T4/A100")

# ==============================================================
# CALLBACK: log limpio por epoch (TF default muestra barras que no siempre se ven en Colab)
# ==============================================================
class EpochLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"  Epoch {epoch+1:02d} — "
              f"loss={logs.get('loss', 0):.4f}  acc={logs.get('accuracy', 0):.4f} | "
              f"val_loss={logs.get('val_loss', 0):.4f}  val_acc={logs.get('val_accuracy', 0):.4f}")

# ==============================================================
# RECOLECTAR RUTAS DE ARCHIVOS Y ETIQUETAS
# ==============================================================
data_dir    = pathlib.Path(RUTA_DATOS)
class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
num_classes = len(class_names)
class_to_idx = {cls: i for i, cls in enumerate(class_names)}
print(f"📊 Clases detectadas ({num_classes}):", class_names)

valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}
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
# PIPELINE DE DATOS TF
# ==============================================================
AUTOTUNE = tf.data.AUTOTUNE

# Augmentation on-the-fly: SOLO se aplica al dataset de train (training=True).
# Se ejecuta sobre la imagen en rango 0-255, ANTES de preprocess_input.
# fill blanco (255) para los huecos geometricos (fondo claro de las muestras).
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.033, fill_mode="constant", fill_value=255.0),      # ~+-12 grados
    layers.RandomTranslation(0.06, 0.06, fill_mode="constant", fill_value=255.0),
    layers.RandomZoom(0.08, fill_mode="constant", fill_value=255.0),           # ~0.92-1.08
    layers.RandomBrightness(0.1, value_range=(0, 255)),
    layers.RandomContrast(0.1),
], name="data_augmentation")

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    return img, label

def make_dataset(file_paths, labels, batch_size, training=False):
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if training:
        ds = ds.shuffle(len(file_paths), seed=SEED)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE).batch(batch_size)
    if training:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

# ==============================================================
# FÁBRICA DE MODELO (se llama cada fold para pesos frescos)
# ==============================================================
def build_model(num_classes):
    print("🔹 Cargando ConvNeXt-Tiny con pesos ImageNet...")
    base_model = ConvNeXtTiny(weights="imagenet", include_top=False,
                              input_shape=(*IMAGE_SIZE, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=output), base_model

# ==============================================================
# PARTICIÓN EN FOLDS
# ==============================================================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
# fold_test_indices[k] contiene los índices asignados al fold k
fold_test_indices = [test_idx for _, test_idx in skf.split(all_files, all_labels)]

# ==============================================================
# K-FOLD LOOP
# fold k: test=fold[k], val=fold[(k+1)%K], train=resto
# ==============================================================
fold_results = []
oof_true, oof_pred = [], []   # predicciones out-of-fold agrupadas (para metricas por clase)

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

    train_ds = make_dataset(all_files[train_indices], all_labels[train_indices],
                            BATCH_SIZE, training=True)
    val_ds   = make_dataset(all_files[val_indices],   all_labels[val_indices],   BATCH_SIZE)
    test_ds  = make_dataset(all_files[test_indices],  all_labels[test_indices],  BATCH_SIZE)

    # Pesos por clase (inverso a la frecuencia) para compensar el desbalance en train
    cw = compute_class_weight("balanced", classes=np.arange(num_classes),
                              y=all_labels[train_indices])
    class_weight = {i: float(w) for i, w in enumerate(cw)}

    # Modelo fresco para cada fold
    tf.random.set_seed(SEED + fold_idx)
    model, base_model = build_model(num_classes)

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.2, patience=3, verbose=1),
        EpochLogger(),
    ]

    # ----------------------------------------------------------
    # FASE 1 — Cabeza solamente
    # ----------------------------------------------------------
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    print("🔒 Fase 1: entrenando solo la cabeza (backbone congelado)...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
              callbacks=callbacks, class_weight=class_weight, verbose=0)

    # ----------------------------------------------------------
    # FASE 2 — Fine-tuning de las últimas 30 capas
    # ----------------------------------------------------------
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    print("🔓 Fase 2: fine-tuning de las últimas 30 capas...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE,
              callbacks=callbacks, class_weight=class_weight, verbose=0)

    # Evaluación en validación (mejor estado restaurado por EarlyStopping)
    _, best_val_acc = model.evaluate(val_ds, verbose=0)
    print(f"✅ Val acc fold {fold_idx}: {best_val_acc:.4f}")

    # Evaluación en test: accuracy + métricas por clase (datos desbalanceados)
    y_true = all_labels[test_indices]
    y_pred = model.predict(test_ds, verbose=0).argmax(axis=1)
    test_acc      = accuracy_score(y_true, y_pred)
    test_f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    test_bal_acc  = balanced_accuracy_score(y_true, y_pred)
    test_cm       = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()
    test_report   = classification_report(y_true, y_pred, labels=list(range(num_classes)),
                                           target_names=class_names, output_dict=True,
                                           zero_division=0)
    print(f"🧪 Test fold {fold_idx}: acc={test_acc:.4f} | "
          f"f1_macro={test_f1_macro:.4f} | bal_acc={test_bal_acc:.4f}")

    fold_results.append({"fold": fold_idx, "val_acc": float(best_val_acc),
                         "test_acc": float(test_acc),
                         "test_f1_macro": float(test_f1_macro),
                         "test_balanced_acc": float(test_bal_acc)})
    # Acumular predicciones out-of-fold (cada imagen es test exactamente una vez)
    oof_true.extend(y_true.tolist())
    oof_pred.extend(y_pred.tolist())

    # Guardar SavedModel y meta del fold
    model.export(fold_output_dir)
    print(f"✅ SavedModel guardado en: {fold_output_dir}")

    meta = {
        "framework":     "tf",
        "arch":          "tf_savedmodel",
        "image_size":    list(IMAGE_SIZE),
        "class_names":   class_names,
        "preprocessing": "convnext",
        "fold":          fold_idx,
        "n_splits":      N_SPLITS,
        "val_acc":       float(best_val_acc),
        "test_acc":      float(test_acc),
        "test_f1_macro":     float(test_f1_macro),
        "test_balanced_acc": float(test_bal_acc),
        "confusion_matrix":      test_cm,
        "classification_report": test_report,
        "notes":         "ConvNeXt-Tiny ImageNet, cabeza 512-ReLU-Dropout(0.4), fine-tuning 30 capas"
    }
    with open(os.path.join(fold_output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# ==============================================================
# RESUMEN CROSS-VALIDATION
# ==============================================================
val_accs  = [r["val_acc"]  for r in fold_results]
test_accs = [r["test_acc"] for r in fold_results]
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
}

# Métricas por clase sobre el OOF agrupado (cada clase evaluada con TODAS sus imágenes)
cv_summary["pooled_oof"] = {
    "n_total":      len(oof_true),
    "accuracy":     float(accuracy_score(oof_true, oof_pred)),
    "f1_macro":     float(f1_score(oof_true, oof_pred, average="macro", zero_division=0)),
    "balanced_acc": float(balanced_accuracy_score(oof_true, oof_pred)),
    "confusion_matrix":      confusion_matrix(oof_true, oof_pred,
                                              labels=list(range(num_classes))).tolist(),
    "classification_report": classification_report(oof_true, oof_pred,
                                                    labels=list(range(num_classes)),
                                                    target_names=class_names,
                                                    output_dict=True, zero_division=0),
}
cv_path = os.path.join(OUTPUT_DIR, "cv_results.json")
with open(cv_path, "w", encoding="utf-8") as f:
    json.dump(cv_summary, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"✅ CV completo — mean_test_acc: {cv_summary['mean_test_acc']:.4f} ± {cv_summary['std_test_acc']:.4f}")
print(f"          — mean_test_f1_macro: {cv_summary['mean_test_f1_macro']:.4f} ± {cv_summary['std_test_f1_macro']:.4f}")
print(f"          — mean_test_balanced_acc: {cv_summary['mean_test_balanced_acc']:.4f} ± {cv_summary['std_test_balanced_acc']:.4f}")
print(f"📊 OOF agrupado ({cv_summary['pooled_oof']['n_total']} imgs) — "
      f"f1_macro: {cv_summary['pooled_oof']['f1_macro']:.4f} | "
      f"balanced_acc: {cv_summary['pooled_oof']['balanced_acc']:.4f}")
print(f"✅ Resultados guardados en: {cv_path}")
print("✅ Entrenamiento ConvNeXt-Tiny completado.")
