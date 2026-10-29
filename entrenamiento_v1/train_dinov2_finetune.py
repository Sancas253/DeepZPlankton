# ==============================================================
# FASE 2: Fine-tuning DINOv2-base — PyTorch + HF
# Google Colab GPU — continúa desde los fold_k/model.pt de
# train_dinov2_mlp.py (backbone congelado + cabeza MLP).
#
# Descongelamos los últimos N_BLOCKS_UNFREEZE bloques transformer
# del encoder (de 12 totales) con LR diferencial por grupo:
#   - cabeza MLP:        LR_HEAD_FT
#   - bloques backbone:  LR_BACKBONE_FT  (<< mucho más bajo)
#
# Esto es distinto a Swin (descongelaba etapas jerárquicas con
# un único LR) y a ConvNeXt (capas por count, único LR).
# El LR diferencial es la forma estándar de fine-tunear ViTs
# sin destruir representaciones self-supervised.
#
# OOF pooled al final (igual que train_dinov2_mlp.py).
# Salida: fold_k/model_ft.pt + fold_k/meta_ft.json + cv_results_ft.json
# ==============================================================
# !pip install transformers scikit-learn -q
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
from transformers import Dinov2Model
from tqdm import tqdm

# ==============================================================
# CONFIGURACIÓN — tiene que coincidir con train_dinov2_mlp.py
# ==============================================================
RUTA_DATOS  = "/content/dataset_local"
INPUT_DIR   = "/content/dinov2_mlp"       # donde están los fold_k/model.pt
OUTPUT_DIR  = "/content/dinov2_finetune"
IMAGE_SIZE  = 224
BATCH_SIZE  = 32
SEED        = 42
N_SPLITS    = 5
NUM_WORKERS = 2

# Fase 2
EPOCHS_FINETUNE   = 15
LR_HEAD_FT        = 1e-4
LR_BACKBONE_FT    = 1e-5
N_BLOCKS_UNFREEZE = 4    # últimos 4 de 12 bloques del encoder DINOv2-base
PATIENCE_FT       = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"🚀 GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No GPU detectada — corre en CPU.")
    print("   → En Colab: Runtime → Cambiar tipo de entorno de ejecución → GPU T4/A100")

# ==============================================================
# TRANSFORMS — iguales a train_dinov2_mlp.py
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
# DATASET Y PARTICIÓN — mismo seed/KFold que train_dinov2_mlp.py
# para que los splits train/val/test sean idénticos por fold
# ==============================================================
full_dataset = datasets.ImageFolder(RUTA_DATOS)
class_names  = full_dataset.classes
num_classes  = len(class_names)
print(f"📊 Clases ({num_classes}):", class_names)
print(f"📊 Total imágenes: {len(full_dataset)}")

all_labels  = np.array(full_dataset.targets)
all_indices = np.arange(len(full_dataset))

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
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
# ARQUITECTURA — igual a DINOv2MLP en train_dinov2_mlp.py
# ==============================================================
class DINOv2MLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-base")
        for param in self.backbone.parameters():
            param.requires_grad = False
        hidden_size = self.backbone.config.hidden_size  # 768
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        cls_token = self.backbone(pixel_values=x).last_hidden_state[:, 0]
        return self.head(cls_token)

# ==============================================================
# K-FOLD LOOP — Fase 2
# ==============================================================
fold_results = []
oof_true, oof_pred = [], []

for fold_idx in range(N_SPLITS):
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx + 1}/{N_SPLITS}")
    print(f"{'='*60}")

    fold_input_dir  = os.path.join(INPUT_DIR,  f"fold_{fold_idx}")
    fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_output_dir, exist_ok=True)

    checkpoint_path = os.path.join(fold_input_dir, "model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No se encontró {checkpoint_path}. "
            "Corré primero train_dinov2_mlp.py hasta completar todos los folds."
        )

    # Splits idénticos a Fase 1
    test_fold   = fold_idx
    val_fold    = (fold_idx + 1) % N_SPLITS
    train_folds = [i for i in range(N_SPLITS) if i != test_fold and i != val_fold]

    test_indices  = fold_test_indices[test_fold]
    val_indices   = fold_test_indices[val_fold]
    train_indices = np.concatenate([fold_test_indices[i] for i in train_folds])
    print(f"  Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")

    train_ds = TransformSubset(Subset(full_dataset, train_indices), train_tf)
    val_ds   = TransformSubset(Subset(full_dataset, val_indices),   val_tf)
    test_ds  = TransformSubset(Subset(full_dataset, test_indices),  val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Cargo el modelo entrenado en Fase 1 y lo muevo al dispositivo
    torch.manual_seed(SEED + fold_idx)
    print("🔹 Cargando DINOv2-base desde HuggingFace...")
    model = DINOv2MLP(num_classes).to(device)
    print(f"  → Cargando pesos Fase 1 desde: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Leer val_acc de Fase 1 para saber de dónde partimos
    meta1_path = os.path.join(fold_input_dir, "meta.json")
    phase1_val_acc = None
    if os.path.exists(meta1_path):
        with open(meta1_path) as f:
            phase1_val_acc = json.load(f).get("val_acc")
        print(f"  → val_acc Fase 1: {phase1_val_acc:.4f}")

    # ----------------------------------------------------------
    # Descongelar los últimos N_BLOCKS_UNFREEZE bloques del encoder.
    # Dos grupos de LR: backbone << cabeza (LR diferencial).
    # ----------------------------------------------------------
    for block in model.backbone.encoder.layer[-N_BLOCKS_UNFREEZE:]:
        for param in block.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"📐 Entrenables: {trainable:,} / {total_p:,}  "
          f"(bloques {12-N_BLOCKS_UNFREEZE+1}–12 + cabeza)")

    print(f"🔓 Fase 2: últimos {N_BLOCKS_UNFREEZE} bloques  "
          f"LR backbone={LR_BACKBONE_FT:.0e} / LR cabeza={LR_HEAD_FT:.0e}")

    cw = compute_class_weight("balanced", classes=np.arange(num_classes),
                              y=all_labels[train_indices])
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam([
        {"params": model.head.parameters(),
         "lr": LR_HEAD_FT},
        {"params": [p for b in model.backbone.encoder.layer[-N_BLOCKS_UNFREEZE:]
                    for p in b.parameters()],
         "lr": LR_BACKBONE_FT},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.2, patience=3)

    best_val_acc = phase1_val_acc if phase1_val_acc else 0.0
    best_state   = None
    patience_ctr = 0

    for epoch in range(1, EPOCHS_FINETUNE + 1):
        model.train()
        tr_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS_FINETUNE} train",
                    leave=False, unit="batch")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total   += imgs.size(0)
            pbar.set_postfix(loss=f"{tr_loss/total:.4f}", acc=f"{correct/total:.4f}")
        tr_loss /= total
        tr_acc   = correct / total

        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{EPOCHS_FINETUNE}   val",
                    leave=False, unit="batch")
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
        print(f"  Epoch {epoch:02d}/{EPOCHS_FINETUNE} — "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE_FT:
                print(f"⏹ Early stopping en epoch {epoch}.")
                break

    if best_state:
        model.load_state_dict(best_state)
    print(f"✅ Mejor val_acc fold {fold_idx} (Fase 2): {best_val_acc:.4f}")

    # Evaluación en test
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
                         "test_balanced_acc": float(test_bal_acc),
                         "phase1_val_acc": phase1_val_acc})
    oof_true.extend(test_true)
    oof_pred.extend(test_preds)

    # Guardar modelo y meta
    weights_path = os.path.join(fold_output_dir, "model_ft.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"✅ Pesos guardados en: {weights_path}")

    meta = {
        "framework":     "pt",
        "arch":          "dinov2_finetune",
        "image_size":    [IMAGE_SIZE, IMAGE_SIZE],
        "class_names":   class_names,
        "preprocessing": "imagenet_meanstd",
        "fold":          fold_idx,
        "n_splits":      N_SPLITS,
        "val_acc":       best_val_acc,
        "phase1_val_acc": phase1_val_acc,
        "test_acc":      test_acc,
        "test_f1_macro":      float(test_f1_macro),
        "test_balanced_acc":  float(test_bal_acc),
        "confusion_matrix":       test_cm,
        "classification_report":  test_report,
        "notes": (f"DINOv2-base: cargado desde Fase1 (backbone congelado + MLP), "
                  f"Fase2 fine-tuning últimos {N_BLOCKS_UNFREEZE} bloques transformer "
                  f"con LR diferencial (cabeza {LR_HEAD_FT:.0e} / backbone {LR_BACKBONE_FT:.0e})"),
    }
    with open(os.path.join(fold_output_dir, "meta_ft.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# ==============================================================
# RESUMEN CROSS-VALIDATION CON OOF POOLED
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

cv_path = os.path.join(OUTPUT_DIR, "cv_results_ft.json")
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
print("✅ Fine-tuning DINOv2 completado.")
