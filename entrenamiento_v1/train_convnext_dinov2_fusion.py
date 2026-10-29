# ==============================================================
# ENTRENAMIENTO Fusión ConvNeXt-Tiny + DINOv2-base — PyTorch
# Google Colab GPU — K-Fold Cross-Validation
# ==============================================================
# Ambos backbones congelados.
# Se concatenan los embeddings y se entrena solo la cabeza MLP.
# Salida: fold_k/model.pt + fold_k/meta.json + cv_results.json
# ==============================================================
# Instalar dependencias en Colab si es necesario:
#   !pip install timm transformers scikit-learn -q
# ==============================================================
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             confusion_matrix, classification_report)
import timm
from transformers import Dinov2Model
from tqdm import tqdm

# ==============================================================
# CONFIGURACIÓN
# ==============================================================
RUTA_DATOS  = "/content/dataset_local"
OUTPUT_DIR  = "/content/convnext_dinov2_fusion"
IMAGE_SIZE  = 224
BATCH_SIZE  = 16
SEED        = 42
EPOCHS      = 30
LR          = 1e-3
N_SPLITS    = 5
PATIENCE    = 7

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Dispositivo: {device}")

# ==============================================================
# TRANSFORMS
# Misma normalización ImageNet para ambos backbones
# ==============================================================
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=12, translate=(0.06, 0.06),
                            scale=(0.92, 1.08), fill=255),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.7))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])
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
# ConvNeXt-Tiny: extrae features espaciales (avg-pooled)
# DINOv2-base:   extrae el token CLS
# Ambos congelados — solo la cabeza MLP es entrenable
# ==============================================================
class ConvNeXtDINOv2Fusion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.convnext = timm.create_model(
            "convnext_tiny", pretrained=True, num_classes=0, global_pool="avg"
        )
        for param in self.convnext.parameters():
            param.requires_grad = False
        convnext_dim = self.convnext.num_features  # 768

        self.dinov2 = Dinov2Model.from_pretrained("facebook/dinov2-base")
        for param in self.dinov2.parameters():
            param.requires_grad = False
        dino_dim = self.dinov2.config.hidden_size  # 768

        fused_dim = convnext_dim + dino_dim  # 1536
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        c = self.convnext(x)
        d = self.dinov2(pixel_values=x).last_hidden_state[:, 0]
        return self.head(torch.cat([c, d], dim=1))

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
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"  Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")

    # Modelo fresco para cada fold
    torch.manual_seed(SEED + fold_idx)
    print("🔹 Cargando ConvNeXt-Tiny (timm) y DINOv2-base (HuggingFace)...")
    model = ConvNeXtDINOv2Fusion(num_classes).to(device)
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

    print("🔒 Entrenando cabeza MLP (ambos backbones congelados)...")
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

    # Guardar modelo y meta del fold
    weights_path = os.path.join(fold_output_dir, "model.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"✅ Pesos guardados en: {weights_path}")

    meta = {
        "framework":     "pt",
        "arch":          "convnext_dinov2_fusion",
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
        "notes":         "Fusión ConvNeXt-Tiny + DINOv2-base congelados, cabeza MLP 1536→512→N"
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
print("✅ Entrenamiento Fusión ConvNeXt+DINOv2 completado.")
