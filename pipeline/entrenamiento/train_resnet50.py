# ==============================================================
# ENTRENAMIENTO ResNet50 (congelado) + MLP — PyTorch + torchvision
# Google Colab GPU — K-Fold Cross-Validation
# ==============================================================
# Réplica de la exploración inicial (M01, Keras) dentro del esquema
# común: ResNet50 de ImageNet completamente congelado y una sola fase
# en la que se entrena solo la cabeza
#   GlobalAveragePooling → Dense(512, ReLU) → Dropout → Dense(N).
# Misma cabeza que DINOv2-MLP: así la comparación aísla el backbone.
# Salida: fold_k/model.pt + fold_k/meta.json + cv_results.json
# ==============================================================
# Instalar dependencias en Colab si es necesario:
#   !pip install scikit-learn -q
# ==============================================================
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             confusion_matrix, classification_report)
from tqdm import tqdm


# ==============================================================
# CONFIGURACIÓN
# ==============================================================
RUTA_DATOS  = "/content/dataset_local"
OUTPUT_DIR  = "/content/resnet50_mlp"
IMAGE_SIZE  = 224
BATCH_SIZE  = 32
SEED        = 42
EPOCHS      = 30
LR          = 1e-3
DROPOUT     = 0.3
N_SPLITS    = 5
PATIENCE    = 7
NUM_WORKERS = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"🚀 GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No GPU detectada — PyTorch corre en CPU.")
    print("   → En Colab: Runtime → Cambiar tipo de entorno de ejecución → GPU T4/A100")
print(f"📁 Dataset: {RUTA_DATOS}")
print(f"📁 Output: {OUTPUT_DIR}")

# ==============================================================
# TRANSFORMS
# ResNet50 fue entrenado con ImageNet mean/std
# ==============================================================
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

# Augmentación predeterminada del pipeline (aumentacion.py, compartida por
# todos los modelos). Solo entrenamiento; val/test van sin augmentar.
import aumentacion
train_tf = aumentacion.transforms_entrenamiento(IMAGE_SIZE, imagenet_mean, imagenet_std)
val_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

# ==============================================================
# DATASET COMPLETO Y PARTICIÓN EN FOLDS
# ==============================================================
full_dataset = datasets.ImageFolder(RUTA_DATOS)
class_names  = full_dataset.classes
num_classes  = len(class_names)
print(f"📊 Clases detectadas ({num_classes}):", class_names)
print(f"📊 Total imágenes: {len(full_dataset)}")

all_labels  = np.array(full_dataset.targets)
all_indices = np.arange(len(full_dataset))

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
# fold_test_indices[k] contiene los índices asignados al fold k
fold_test_indices = [test_idx for _, test_idx in skf.split(all_indices, all_labels)]

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label

# ==============================================================
# MODELO
# Backbone congelado; solo la cabeza MLP es entrenable.
# ==============================================================
class ResNet50MLP(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        pesos = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=pesos)
        self.backbone.fc = nn.Identity()   # sale del GlobalAveragePooling: 2048
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, num_classes),
        )

    def train(self, mode=True):
        # ResNet tiene BatchNorm: en modo train actualizaría sus medias y
        # varianzas aunque los pesos estén congelados. El backbone queda
        # siempre en eval (como trainable=False en Keras).
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        return self.head(self.backbone(x))

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

    train_ds = TransformSubset(Subset(full_dataset, train_indices), train_tf)
    val_ds   = TransformSubset(Subset(full_dataset, val_indices),   val_tf)
    test_ds  = TransformSubset(Subset(full_dataset, test_indices),  val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print(f"  Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")

    # Modelo fresco para cada fold
    torch.manual_seed(SEED + fold_idx)
    print("🔹 Cargando ResNet50 (ImageNet) desde torchvision...")
    model = ResNet50MLP(num_classes).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"📐 Parámetros entrenables: {trainable:,} / {total_p:,}")

    # Pesos por clase (inverso a la frecuencia) para compensar el desbalance en train
    cw = compute_class_weight("balanced", classes=np.arange(num_classes),
                              y=all_labels[train_indices])
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.head.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.2, patience=3)

    best_val_acc = 0.0
    best_state   = None
    patience_ctr = 0

    print("🔒 Entrenando cabeza MLP (ResNet50 completamente congelado)...")
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        tr_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} train", leave=False, unit="batch")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            tr_loss  += loss.item() * imgs.size(0)
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += imgs.size(0)
            pbar.set_postfix(loss=f"{tr_loss/total:.4f}", acc=f"{correct/total:.4f}")
        tr_loss /= total
        tr_acc   = correct / total

        # Val
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{EPOCHS}   val", leave=False, unit="batch")
        with torch.no_grad():
            for imgs, labels in pbar:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                va_loss    += loss.item() * imgs.size(0)
                va_correct += (logits.argmax(1) == labels).sum().item()
                va_total   += imgs.size(0)
                pbar.set_postfix(loss=f"{va_loss/va_total:.4f}", acc=f"{va_correct/va_total:.4f}")
        va_loss /= va_total
        va_acc   = va_correct / va_total

        scheduler.step(va_loss)
        print(f"Epoch {epoch:02d}/{EPOCHS} — "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"⏹ Early stopping en epoch {epoch}.")
                break

    if best_state:
        model.load_state_dict(best_state)
    print(f"✅ Mejor val_acc fold {fold_idx}: {best_val_acc:.4f}")

    # Evaluación en test: accuracy + métricas por clase (datos desbalanceados)
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            logits = model(imgs.to(device))
            test_preds.extend(logits.argmax(1).cpu().numpy().tolist())
            test_true.extend(labels.numpy().tolist())
    test_acc      = accuracy_score(test_true, test_preds)
    test_f1_macro = f1_score(test_true, test_preds, average="macro", zero_division=0)
    test_bal_acc  = balanced_accuracy_score(test_true, test_preds)
    test_cm       = confusion_matrix(test_true, test_preds,
                                     labels=list(range(num_classes))).tolist()
    test_report   = classification_report(test_true, test_preds,
                                          labels=list(range(num_classes)),
                                          target_names=class_names, output_dict=True,
                                          zero_division=0)
    print(f"🧪 Test fold {fold_idx}: acc={test_acc:.4f} | "
          f"f1_macro={test_f1_macro:.4f} | bal_acc={test_bal_acc:.4f}")

    fold_results.append({"fold": fold_idx, "val_acc": best_val_acc, "test_acc": test_acc,
                         "test_f1_macro": float(test_f1_macro),
                         "test_balanced_acc": float(test_bal_acc)})
    # Acumular predicciones out-of-fold (cada imagen es test exactamente una vez)
    oof_true.extend(test_true)
    oof_pred.extend(test_preds)
    # Predicciones por imagen del test (-> oof_predicciones.csv) y split.json
    import predicciones
    predicciones.guardar_fold(model, test_loader, full_dataset, RUTA_DATOS, class_names,
                              fold_output_dir, device, train_indices, val_indices, test_indices)

    # Guardar modelo y meta del fold
    weights_path = os.path.join(fold_output_dir, "model.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"✅ Pesos guardados en: {weights_path}")

    meta = {
        "framework":     "pt",
        "arch":          "resnet50_mlp",
        "image_size":    [IMAGE_SIZE, IMAGE_SIZE],
        "class_names":   class_names,
        "preprocessing": "imagenet_meanstd",
        "fold":          fold_idx,
        "n_splits":      N_SPLITS,
        "val_acc":       best_val_acc,
        "test_acc":      test_acc,
        "test_f1_macro":     float(test_f1_macro),
        "test_balanced_acc": float(test_bal_acc),
        "confusion_matrix":      test_cm,
        "classification_report": test_report,
        "notes":         f"ResNet50 ImageNet congelado, cabeza MLP 2048→512-ReLU-Dropout({DROPOUT})→N"
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
print("✅ Entrenamiento ResNet50-MLP completado.")
