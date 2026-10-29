# Fusión en paralelo: ConvNeXt-Tiny completo (receta convnext_tiny_plus, 288px)
# y DINOv2 ViT-B/14 (receta dinov2_ft, 224px) miran la imagen original, ambos
# con fine-tuning. Se concatenan pooling de ConvNeXt + CLS del ViT -> Linear.
# Primeras ADAPTER_EPOCHS: solo la cabeza; después se entrena todo.
# Receta plus: AdamW, layer decay, warmup/coseno, EMA y weighted smooth CE.
# Mismos folds/dataset/aumentacion/val_acc que el resto del pipeline.
# ADAPTER_EPOCHS esta incluido en EPOCHS; luego se ajusta todo el Tiny.
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
from modelo_fusion_plus import ConvNeXtDINOv2FusionPlus
from timm.scheduler import CosineLRScheduler
from timm.utils import ModelEmaV2
from tqdm import tqdm

# ==============================================================
# CONFIGURACIÓN
# ==============================================================
RUTA_DATOS      = "/content/dataset_local"
OUTPUT_DIR      = "/content/convnext_dinov2_fusion_plus"
PESOS           = 'convnext_tiny.fb_in22k_ft_in1k'
PESOS_VIT       = 'vit_base_patch14_dinov2.lvd142m'
CLASS_WEIGHTS   = "balanced"    # "balanced" o "sqrt"
IMAGE_SIZE      = 288         # entrada y ConvNeXt (como convnext_tiny_plus)
VIT_SIZE        = 224         # el ViT reescala adentro (como dinov2_ft)
BATCH_SIZE      = 16          # dos backbones completos con gradientes
SEED            = 42
EPOCHS          = 30
WARMUP_EPOCHS   = 3
LR              = 1e-4
LR_NEW          = 1e-3
ADAPTER_EPOCHS  = 3
LR_MIN          = 1e-6
WEIGHT_DECAY    = 0.05
LAYER_DECAY     = 0.8         # igual que convnext_tiny_plus
VIT_LR          = 5e-5        # igual que dinov2_ft
VIT_LAYER_DECAY = 0.75
LABEL_SMOOTHING = 0.1
DROP_PATH       = 0.1
EMA_DECAY       = 0.999
N_SPLITS        = 5
PATIENCE        = 8
NUM_WORKERS     = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"
print(f"🚀 Dispositivo: {device} | precisión mixta: {USE_AMP}")
# Version de arquitectura para distinguir los checkpoints anteriores
ARCH = "convnext_dinov2_fusion_plus"
print(f"🔹 Pesos: {PESOS} | {IMAGE_SIZE}px | LR {LR} | layer decay {LAYER_DECAY} | "
      f"class weights {CLASS_WEIGHTS}")

# ==============================================================
# TRANSFORMS
# ImageNet mean/std (lo que espera ConvNeXt-Tiny de timm)
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

if N_SPLITS < 3 or np.bincount(all_labels).min() < N_SPLITS:
    raise ValueError("N_SPLITS >= 3 y al menos N_SPLITS imagenes por clase")
if EPOCHS < 1 or BATCH_SIZE < 1 or not 0 <= ADAPTER_EPOCHS < EPOCHS:
    raise ValueError("EPOCHS/BATCH_SIZE positivos y 0 <= ADAPTER_EPOCHS < EPOCHS")

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
# LOSS: class weights + label smoothing
# No se usa nn.CrossEntropyLoss(weight=..., label_smoothing=...): PyTorch
# pondera también el término de suavizado con el peso de CADA clase, así que
# toda imagen termina empujada hacia las clases raras (pesos altos) y la
# mayoritaria (Calanoida, peso bajo) se deja de aprender. Acá cada imagen
# usa solo el peso de su clase real en los dos términos.
# ==============================================================
class WeightedSmoothCE(nn.Module):
    def __init__(self, weight, smoothing):
        super().__init__()
        self.register_buffer("weight", weight)
        self.smoothing = smoothing

    def forward(self, logits, target):
        logp = torch.log_softmax(logits.float(), dim=1)
        nll    = -logp.gather(1, target[:, None]).squeeze(1)
        smooth = -logp.mean(dim=1)
        loss   = (1 - self.smoothing) * nll + self.smoothing * smooth
        w = self.weight[target]
        return (w * loss).sum() / w.sum()

# ==============================================================
# HELPERS DE ENTRENAMIENTO
# ==============================================================
def train_epoch(model, ema, loader, optimizer, scaler, criterion, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{EPOCHS} train", leave=False, unit="batch")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")
    return total_loss / total, correct / total

def predecir(model, loader, desc):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=desc, leave=False, unit="batch"):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
                logits = model(imgs.to(device))
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            true.extend(labels.numpy().tolist())
    return true, preds

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

    if len(train_indices) < BATCH_SIZE:
        raise ValueError("Train menor que BATCH_SIZE; reducir BATCH_SIZE para drop_last=True")
    split = {name: [os.path.relpath(full_dataset.samples[i][0], RUTA_DATOS)
                    for i in indices]
             for name, indices in (("train", train_indices), ("val", val_indices),
                                    ("test", test_indices))}
    with open(os.path.join(fold_output_dir, "split.json"), "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    train_ds = TransformSubset(Subset(full_dataset, train_indices), train_tf)
    val_ds   = TransformSubset(Subset(full_dataset, val_indices),   val_tf)
    test_ds  = TransformSubset(Subset(full_dataset, test_indices),  val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print(f"  Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")

    # Modelo fresco para cada fold
    torch.manual_seed(SEED + fold_idx)
    print(f"🔹 Cargando {PESOS}...")
    model = ConvNeXtDINOv2FusionPlus(num_classes, image_size=IMAGE_SIZE, vit_size=VIT_SIZE,
                                     pretrained=True, drop_path_rate=DROP_PATH,
                                     vit_drop_path_rate=DROP_PATH, vit_weights=PESOS_VIT,
                                     classifier_weights=PESOS)
    model.to(device)
    ema = ModelEmaV2(model, decay=EMA_DECAY)

    # Pesos por clase (inverso a la frecuencia) para compensar el desbalance en train
    cw = compute_class_weight("balanced", classes=np.arange(num_classes),
                              y=all_labels[train_indices])
    if CLASS_WEIGHTS == "sqrt":
        cw = np.sqrt(cw)
    criterion = WeightedSmoothCE(torch.tensor(cw, dtype=torch.float32, device=device),
                                 LABEL_SMOOTHING)

    optimizer = model.make_optimizer(LR, LR_NEW, WEIGHT_DECAY, LAYER_DECAY,
                                     VIT_LR, VIT_LAYER_DECAY)
    scheduler = CosineLRScheduler(optimizer, t_initial=EPOCHS, lr_min=LR_MIN,
                                  warmup_t=WARMUP_EPOCHS, warmup_lr_init=LR_MIN)
    scaler = torch.amp.GradScaler(device.type, enabled=USE_AMP)

    best_val_acc = -1.0
    best_state   = None
    patience_ctr = 0

    print("🔓 Primero solo la cabeza; luego fine-tuning de ConvNeXt y DINOv2 en paralelo.")
    for epoch in range(1, EPOCHS + 1):
        model.set_adapter_only(epoch <= ADAPTER_EPOCHS)
        tr_loss, tr_acc = train_epoch(model, ema, train_loader, optimizer, scaler,
                                      criterion, epoch)
        scheduler.step(epoch)

        # Se valida el modelo EMA, que es el que se guarda
        va_true, va_pred = predecir(ema.module, val_loader, f"Epoch {epoch:02d}/{EPOCHS}   val")
        va_acc = accuracy_score(va_true, va_pred)
        va_f1  = f1_score(va_true, va_pred, average="macro", zero_division=0)
        lr_actual = max(g["lr"] for g in optimizer.param_groups)
        print(f"Epoch {epoch:02d}/{EPOCHS} — lr={lr_actual:.2e} | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_acc(EMA)={va_acc:.4f} val_f1_macro(EMA)={va_f1:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone() for k, v in ema.module.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"⏹ Early stopping en epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    print(f"✅ Mejor val_acc fold {fold_idx}: {best_val_acc:.4f}")
    del ema

    # Evaluación en test: accuracy + métricas por clase (datos desbalanceados)
    test_true, test_preds = predecir(model, test_loader, "test")
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

    fold_results.append({"fold": fold_idx, "val_acc": float(best_val_acc), "test_acc": test_acc,
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
        "arch":          ARCH,
        "image_size":    [IMAGE_SIZE, IMAGE_SIZE],
        "class_names":   class_names,
        "preprocessing": "imagenet_meanstd",
        "fold":          fold_idx,
        "n_splits":      N_SPLITS,
        "val_acc":       float(best_val_acc),
        "test_acc":      test_acc,
        "test_f1_macro":     float(test_f1_macro),
        "test_balanced_acc": float(test_bal_acc),
        "confusion_matrix":      test_cm,
        "classification_report": test_report,
        "notes": (f"Fusión paralela: DINOv2 {PESOS_VIT} a {VIT_SIZE}px vit_lr={VIT_LR} vit_layer_decay={VIT_LAYER_DECAY} + ConvNeXt completo: {PESOS}, {IMAGE_SIZE}px, class weights "
                  f"{CLASS_WEIGHTS}, adapter_epochs={ADAPTER_EPOCHS}, lr_new={LR_NEW}; fine-tuning Tiny "
                  f"AdamW lr={LR} layer_decay={LAYER_DECAY}, coseno, label smoothing "
                  f"{LABEL_SMOOTHING}, drop path {DROP_PATH}, EMA {EMA_DECAY}"),
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
print("✅ Entrenamiento fusión ConvNeXt-Tiny + DINOv2 (paralelo) completado.")
