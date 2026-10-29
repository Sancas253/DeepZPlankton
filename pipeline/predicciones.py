"""
predicciones.py — predicciones por imagen del test de cada fold.

Cada script de entrenamiento llama a guardar_fold (PyTorch) o a
guardar_probabilidades (TF) al evaluar el test de un fold. Quedan:
  fold_k/predicciones_test.csv   una fila por imagen de test del fold:
                                 ruta, real, predicha, confianza, acierto, p_<clase>...
  fold_k/split.json              rutas de train/val/test (si el script no lo guardó)
y unir_oof junta los folds en <salida>/oof_predicciones.csv: como cada imagen
es test en exactamente un fold, queda una fila por imagen del dataset, predicha
por el modelo que no la vio. Es la base de comparar_corridas.py (McNemar,
superposición de errores) y de analizar_modelo.py --solo-test.
"""
import csv
import glob
import json
import os

import numpy as np

COLUMNAS = ["ruta", "real", "predicha", "confianza", "acierto"]


def _rel(ruta, raiz):
    return os.path.relpath(ruta, raiz).replace("\\", "/")


def guardar_probabilidades(probs, rutas, reales, class_names, fold_dir, split=None):
    """probs: N x C (softmax), rutas relativas al dataset, reales: índices de clase."""
    probs = np.asarray(probs, dtype=np.float64)
    os.makedirs(fold_dir, exist_ok=True)
    with open(os.path.join(fold_dir, "predicciones_test.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNAS + [f"p_{c}" for c in class_names])
        for ruta, real, p in zip(rutas, reales, probs):
            pred = int(p.argmax())
            w.writerow([ruta, class_names[int(real)], class_names[pred], f"{p[pred]:.6f}",
                        int(pred == int(real))] + [f"{x:.6f}" for x in p])
    ruta_split = os.path.join(fold_dir, "split.json")
    if split is not None and not os.path.exists(ruta_split):
        with open(ruta_split, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)


def armar_split(rutas_de, train_indices, val_indices, test_indices):
    return {nombre: [rutas_de(i) for i in idx] for nombre, idx in
            (("train", train_indices), ("val", val_indices), ("test", test_indices))}


def guardar_fold(model, test_loader, full_dataset, raiz, class_names, fold_dir, device,
                 train_indices, val_indices, test_indices):
    """Versión PyTorch: corre el test (loader sin shuffle, en el orden de test_indices)."""
    import torch
    rutas_de = lambda i: _rel(full_dataset.samples[i][0], raiz)
    model.eval()
    probs = []
    with torch.no_grad():
        for imgs, _ in test_loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(imgs.to(device))
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    reales = [full_dataset.samples[i][1] for i in test_indices]
    guardar_probabilidades(np.concatenate(probs), [rutas_de(i) for i in test_indices], reales,
                           class_names, fold_dir,
                           armar_split(rutas_de, train_indices, val_indices, test_indices))


def unir_oof(salida):
    """Junta fold_*/predicciones_test.csv en <salida>/oof_predicciones.csv (con columna fold)."""
    archivos = sorted(glob.glob(os.path.join(salida, "fold_*", "predicciones_test.csv")),
                      key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[1]))
    if not archivos:
        return None
    destino = os.path.join(salida, "oof_predicciones.csv")
    with open(destino, "w", newline="", encoding="utf-8") as out:
        w = None
        for ruta in archivos:
            fold = os.path.basename(os.path.dirname(ruta)).split("_")[1]
            with open(ruta, encoding="utf-8") as f:
                r = csv.reader(f)
                cabecera = next(r)
                if w is None:
                    w = csv.writer(out)
                    w.writerow(["fold"] + cabecera)
                for fila in r:
                    w.writerow([fold] + fila)
    return destino


def leer_oof(ruta):
    """Lee un oof_predicciones.csv (o la carpeta que lo contiene) -> dict ruta -> fila."""
    if os.path.isdir(ruta):
        ruta = os.path.join(ruta, "oof_predicciones.csv")
    with open(ruta, encoding="utf-8") as f:
        return {fila["ruta"]: fila for fila in csv.DictReader(f)}
