# ==============================================================
# ENTRENAMIENTO Swin-Tiny — PyTorch + timm — PC LOCAL (Windows)
# Versión RÁPIDA: un solo fold (sin ensemble de 5)
# ==============================================================
# Fase 1: backbone congelado, solo cabeza
# Fase 2: fine-tuning de las últimas 2 etapas del backbone
# Salida: fold_0/model.pt + fold_0/meta.json + cv_results.json
# ==============================================================
# Igual que train_swin_tiny_local.py pero entrena un solo modelo
# (train/val/test únicos, proporción 60/20/20) en vez de los 5
# folds del ensemble. ~5x más rápido, a costa de perder el
# ensemble y la robustez de la validación cruzada.
# average_and_export_onnx.py también funciona con esta salida:
# el "ensemble" queda de un solo modelo (fold_0.onnx).
#
# Instalar dependencias si hace falta:
#   pip install torch torchvision timm scikit-learn tqdm
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
from tqdm import tqdm

# ==============================================================
# CONFIGURACIÓN
# ==============================================================
# Carpeta de Windows con el dataset, organizado como ImageFolder
# (una subcarpeta por clase, con las imágenes adentro). Cambiar
# esta ruta por la carpeta real.
RUTA_DATOS      = r"C:\Users\santi\Escritorio\dataset"
OUTPUT_DIR      = r"C:\Users\santi\Escritorio\swin_tiny_finetuned_rapido"
IMAGE_SIZE      = 224
BATCH_SIZE      = 32
SEED            = 42
EPOCHS_HEAD     = 15
EPOCHS_FINETUNE = 20
LR_HEAD         = 1e-3
LR_FINETUNE     = 1e-5
# N_SPLITS solo define la proporción del split (1/5 = 20% test,
# 20% val, 60% train, estratificado). No se entrenan los 5 folds:
# más abajo el loop corta después del primero.
N_SPLITS        = 5
PATIENCE        = 5
# En Windows, num_workers>0 obliga a que TODO el código quede detrás
# del "if __name__ == '__main__':" (más abajo) porque cada worker
# vuelve a importar este archivo. Con 0 se evita ese problema, a
# costa de cargar imágenes en el proceso principal (más lento).
NUM_WORKERS     = 0

# ==============================================================
# TRANSFORMS
# ImageNet mean/std (lo que espera Swin-Tiny de timm)
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
# HELPERS DE ENTRENAMIENTO
# ==============================================================
def train_epoch(model, loader, optimizer, criterion, epoch, epochs, label, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"[{label}] Epoch {epoch:02d}/{epochs} train",
                leave=False, unit="batch")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")
    return total_loss / total, correct / total

def eval_epoch(model, loader, criterion, epoch, epochs, label, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"[{label}] Epoch {epoch:02d}/{epochs}   val",
                leave=False, unit="batch")
    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
            pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")
    return total_loss / total, correct / total

def run_training(model, loader_train, loader_val, optimizer, scheduler,
                 criterion, epochs, label, device):
    best_val_acc = 0.0
    best_state   = None
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, loader_train, optimizer, criterion, epoch, epochs, label, device)
        va_loss, va_acc = eval_epoch(model, loader_val, criterion, epoch, epochs, label, device)
        scheduler.step(va_loss)
        print(f"[{label}] Epoch {epoch:02d}/{epochs} — "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  ⏹ Early stopping en epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  ✅ Mejor val_acc [{label}]: {best_val_acc:.4f}")
    return best_val_acc


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Dispositivo: {device}")
    if device.type == "cpu":
        print("⚠️  Entrenando en CPU: va a ser más lento que en GPU (pero es un solo fold).")

    # ==============================================================
    # DATASET COMPLETO Y SPLIT ÚNICO (train/val/test estratificado)
    # ==============================================================
    full_dataset = datasets.ImageFolder(RUTA_DATOS)
    class_names  = full_dataset.classes
    num_classes  = len(class_names)
    print(f"📊 Clases detectadas ({num_classes}):", class_names)
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

    # Un solo split: fold 0 = test, fold 1 = val, resto = train.
    fold_idx      = 0
    fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_output_dir, exist_ok=True)

    test_fold   = 0
    val_fold    = 1
    train_folds = [i for i in range(N_SPLITS) if i != test_fold and i != val_fold]

    test_indices  = fold_test_indices[test_fold]
    val_indices   = fold_test_indices[val_fold]
    train_indices = np.concatenate([fold_test_indices[i] for i in train_folds])

    train_ds = TransformSubset(Subset(full_dataset, train_indices), train_tf)
    val_ds   = TransformSubset(Subset(full_dataset, val_indices),   val_tf)
    test_ds  = TransformSubset(Subset(full_dataset, test_indices),  val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

    print(f"  Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")

    torch.manual_seed(SEED)
    print("🔹 Cargando Swin-Tiny con pesos ImageNet...")
    model = timm.create_model(
        "swin_tiny_patch4_window7_224", pretrained=True, num_classes=num_classes
    )
    model.to(device)

    # Pesos por clase (inverso a la frecuencia) para compensar el desbalance en train
    cw = compute_class_weight("balanced", classes=np.arange(num_classes),
                              y=all_labels[train_indices])
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32, device=device))

    # ----------------------------------------------------------
    # FASE 1 — Congelar backbone, entrenar solo cabeza
    # ----------------------------------------------------------
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("head")

    optimizer1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1, factor=0.2, patience=3)

    print("🔒 Fase 1: entrenando solo la cabeza (backbone congelado)...")
    best_val_acc = run_training(model, train_loader, val_loader, optimizer1, scheduler1,
                                criterion, EPOCHS_HEAD, "Fase1", device)

    # ----------------------------------------------------------
    # FASE 2 — Descongelar últimas 2 etapas (layers.2 y layers.3)
    # ----------------------------------------------------------
    for name, param in model.named_parameters():
        param.requires_grad = (
            name.startswith("layers.2") or
            name.startswith("layers.3") or
            name.startswith("head")
        )

    optimizer2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FINETUNE)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, factor=0.2, patience=3)

    print("🔓 Fase 2: fine-tuning de las últimas 2 etapas...")
    best_val_acc = run_training(model, train_loader, val_loader, optimizer2, scheduler2,
                                criterion, EPOCHS_FINETUNE, "Fase2", device)

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
    print(f"🧪 Test: acc={test_acc:.4f} | f1_macro={test_f1_macro:.4f} | bal_acc={test_bal_acc:.4f}")

    # Guardar modelo y meta
    weights_path = os.path.join(fold_output_dir, "model.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"✅ Pesos guardados en: {weights_path}")

    meta = {
        "framework":     "pt",
        "arch":          "swin_tiny",
        "image_size":    [IMAGE_SIZE, IMAGE_SIZE],
        "class_names":   class_names,
        "preprocessing": "imagenet_meanstd",
        "fold":          fold_idx,
        "n_splits":      1,
        "val_acc":       best_val_acc,
        "test_acc":      test_acc,
        "test_f1_macro":     float(test_f1_macro),
        "test_balanced_acc": float(test_bal_acc),
        "confusion_matrix":      test_cm,
        "classification_report": test_report,
        "notes":         "Swin-Tiny ImageNet, cabeza lineal, fine-tuning etapas 2 y 3 — split único (sin ensemble)"
    }
    with open(os.path.join(fold_output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # cv_results.json con un único resultado, para mantener el mismo
    # formato de salida que la versión con ensemble.
    cv_summary = {
        "n_splits": 1,
        "folds": [{
            "fold": fold_idx, "val_acc": best_val_acc, "test_acc": test_acc,
            "test_f1_macro": float(test_f1_macro),
            "test_balanced_acc": float(test_bal_acc),
        }],
        "mean_test_acc": test_acc,
        "mean_test_f1_macro": float(test_f1_macro),
        "mean_test_balanced_acc": float(test_bal_acc),
    }
    cv_path = os.path.join(OUTPUT_DIR, "cv_results.json")
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, ensure_ascii=False, indent=2)

    print(f"✅ Resultados guardados en: {cv_path}")
    print("✅ Entrenamiento Swin-Tiny (un fold) completado.")


if __name__ == "__main__":
    main()
