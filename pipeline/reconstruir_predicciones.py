"""
reconstruir_predicciones.py — oof_predicciones.csv para entrenamientos viejos.

Los entrenamientos anteriores no guardaban split.json ni las predicciones por
imagen. Como los folds son deterministas (StratifiedKFold con shuffle y
SEED=42 sobre ImageFolder, test=k, val=k+1, train=resto), se reconstruyen a
partir del MISMO dataset limpio y se predice cada imagen de test con el fold
que no la vio. Quedan fold_k/split.json, fold_k/predicciones_test.csv y
oof_predicciones.csv, igual que en un entrenamiento nuevo.

Controles para no reconstruir mal (si fallan, no sigue salvo --forzar):
  - las clases del dataset coinciden con las del meta.json;
  - la cantidad de imágenes por clase coincide con cv_results.json;
  - la accuracy de cada fold reconstruido coincide con test_acc de su meta.json.

Solo modelos PyTorch. Uso (Colab con GPU, el dataset limpio de esa corrida):
  python reconstruir_predicciones.py --modelo /content/drive/MyDrive/.../dinov2_ft \
      --dataset /content/deepz/dataset_limpio
"""
import argparse
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import predicciones  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main(argv=None):
    p = argparse.ArgumentParser(description="Reconstruye las predicciones OOF de un entrenamiento viejo.")
    p.add_argument("--modelo", required=True, help="Carpeta del entrenamiento (fold_k/model.pt + meta.json)")
    p.add_argument("--dataset", required=True, help="El mismo dataset limpio con el que se entrenó")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--forzar", action="store_true", help="Seguir aunque fallen los controles")
    args = p.parse_args(argv)

    import torch
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    from exportar_onnx import build_model

    def control(ok, msg):
        if not ok:
            if not args.forzar:
                raise SystemExit(f"❌ {msg}\n   (--forzar para seguir igual: los folds podrían no ser los del entrenamiento)")
            print(f"⚠️  {msg}")

    with open(os.path.join(args.modelo, "fold_0", "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    clases = meta["class_names"]
    n_splits = meta.get("n_splits", 5)
    alto, ancho = meta["image_size"]
    tf = transforms.Compose([transforms.Resize((alto, ancho)), transforms.ToTensor(),
                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    full = datasets.ImageFolder(args.dataset, transform=tf)
    control(full.classes == clases, f"Las clases del dataset no coinciden con meta.json: "
                                    f"{sorted(set(full.classes) ^ set(clases))}")
    labels = np.array(full.targets)

    ruta_cv = os.path.join(args.modelo, "cv_results.json")
    if os.path.exists(ruta_cv):
        with open(ruta_cv, encoding="utf-8") as f:
            reporte = json.load(f)["pooled_oof"]["classification_report"]
        distintas = [c for k, c in enumerate(clases)
                     if c in reporte and int(reporte[c]["support"]) != int((labels == k).sum())]
        control(not distintas, f"Cantidad de imágenes distinta a la del entrenamiento en: {distintas}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    partes = [test for _, test in skf.split(np.arange(len(labels)), labels)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 {meta['arch']} | {len(full)} imágenes | {n_splits} folds | {device}")

    for k in range(n_splits):
        carpeta = os.path.join(args.modelo, f"fold_{k}")
        if not os.path.exists(os.path.join(carpeta, "model.pt")):
            print(f"  fold_{k}: sin model.pt, se saltea")
            continue
        test_idx, val_idx = partes[k], partes[(k + 1) % n_splits]
        train_idx = np.concatenate([partes[i] for i in range(n_splits) if i not in (k, (k + 1) % n_splits)])
        with open(os.path.join(carpeta, "meta.json"), encoding="utf-8") as f:
            meta_k = json.load(f)
        model = build_model(meta_k["arch"], len(clases), meta_k["image_size"])
        model.load_state_dict(torch.load(os.path.join(carpeta, "model.pt"), map_location="cpu",
                                         weights_only=True))
        model.to(device)
        loader = DataLoader(Subset(full, test_idx), batch_size=args.batch, shuffle=False, num_workers=0)
        predicciones.guardar_fold(model, loader, full, args.dataset, clases, carpeta, device,
                                  train_idx, val_idx, test_idx)
        leida = predicciones.leer_oof(os.path.join(carpeta, "predicciones_test.csv"))
        acc = np.mean([int(r["acierto"]) for r in leida.values()])
        esperado = meta_k.get("test_acc")
        print(f"  fold_{k}: accuracy reconstruida {acc:.4f}"
              + (f" | entrenamiento {esperado:.4f}" if esperado is not None else ""))
        if esperado is not None:
            control(abs(acc - esperado) < 0.003,
                    f"fold_{k}: accuracy {acc:.4f} distinta a la del entrenamiento {esperado:.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"✅ {predicciones.unir_oof(args.modelo)}")


if __name__ == "__main__":
    main()
