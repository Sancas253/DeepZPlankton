"""
gradcam_errores.py — Grad-CAM de las imágenes en las que erra un entrenamiento.

Lee <modelo>/oof_predicciones.csv (cada imagen predicha por el fold que NO la vio)
y, para cada error, carga el model.pt de ESE fold y guarda:
  imagen | Grad-CAM de cada rama (clase predicha) | Grad-CAM de la clase real
  | atención CLS→parches del último bloque (solo modelos con ViT/DINOv2)

Salida:
  <salida>/alta_confianza/      errores con confianza >= --umbral-alta (el modelo
                                "estaba seguro": candidatos a etiqueta mal puesta)
  <salida>/<real>__a__<pred>/   el resto, agrupados por confusión
  <salida>/errores.csv          lista con la carpeta donde quedó cada uno
  <salida>/umap_errores.png     UMAP de analisis/fold_*/embeddings.npz con errores marcados

Ejemplo (Colab, con el dataset limpio descomprimido):
  python gradcam_errores.py \
      --modelo /content/drive/MyDrive/deepz_resultados/<corrida>/convnext_dinov2_fusion_plus \
      --imagenes /content/deepz/dataset_limpio --salida /content/drive/MyDrive/gradcam_errores
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from analizar_modelo import (IMAGENET_MEAN, IMAGENET_STD, AtencionCLS, GradCAM, cargar_modelo,  # noqa: E402
                             graficar_proyeccion, sin_vit, nombre_seguro, proyectar_2d)


def guardar(img, paneles, titulo, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 1 + len(paneles), figsize=(3.6 * (1 + len(paneles)), 4.2))
    axes[0].imshow(img)
    axes[0].set_title("imagen", fontsize=8)
    for ax, (nombre, mapa) in zip(axes[1:], paneles):
        ax.imshow(img)
        ax.imshow(mapa, cmap="jet", alpha=0.45)
        ax.set_title(nombre, fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(titulo, fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=100)
    plt.close(fig)


def umap_errores(modelo, salida, semilla, folds=range(20)):
    X, real, pred, nombres = [], [], [], None
    for k in folds:
        ruta = os.path.join(modelo, "analisis", f"fold_{k}", "embeddings.npz")
        if os.path.exists(ruta):
            d = np.load(ruta, allow_pickle=True)
            X.append(d["X"]), real.append(d["real"]), pred.append(d["pred"])
            nombres = list(d["clases"])
    if not X:
        print("⚠️  No hay analisis/fold_*/embeddings.npz: se saltea el UMAP.")
        return
    X, real, pred = np.concatenate(X), np.concatenate(real), np.concatenate(pred)
    Z, metodo = proyectar_2d(X, semilla)
    graficar_proyeccion(Z, real, nombres, f"{metodo} de las features de test\n"
                        "círculo negro = error", os.path.join(salida, "umap_errores.png"),
                        errores=(real >= 0) & (pred != real))
    print(f"🗺️  {metodo} guardado")


def main(argv=None):
    p = argparse.ArgumentParser(description="Grad-CAM de los errores out-of-fold.")
    p.add_argument("--modelo", required=True, help="Carpeta del entrenamiento (con fold_k/ y oof_predicciones.csv)")
    p.add_argument("--imagenes", required=True, help="Dataset limpio (subcarpetas por clase)")
    p.add_argument("--salida", default="gradcam_errores")
    p.add_argument("--umbral-alta", type=float, default=0.9)
    p.add_argument("--max", type=int, default=0, help="Tope de errores (0 = todos; primero los de más confianza)")
    p.add_argument("--fold", type=int, default=None, help="Solo los errores de este fold (un único modelo)")
    p.add_argument("--sin-umap", action="store_true")
    p.add_argument("--semilla", type=int, default=42)
    args = p.parse_args(argv)

    with open(os.path.join(args.modelo, "oof_predicciones.csv"), encoding="utf-8") as f:
        errores = [r for r in csv.DictReader(f) if r["acierto"] == "0"
                   and (args.fold is None or int(r["fold"]) == args.fold)]
    errores.sort(key=lambda r: -float(r["confianza"]))
    if args.max:
        errores = errores[:args.max]
    print(f"📊 {len(errores)} errores")
    os.makedirs(args.salida, exist_ok=True)
    if not args.sin_umap:
        umap_errores(args.modelo, args.salida, args.semilla,
                     range(20) if args.fold is None else [args.fold])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filas = []
    for fold in sorted({int(r["fold"]) for r in errores}):
        model, meta, _, _ = cargar_modelo(args.modelo, fold)
        model.to(device)
        clases = meta["class_names"]
        alto, ancho = meta["image_size"]
        tf = transforms.Compose([transforms.Resize((alto, ancho)), transforms.ToTensor(),
                                 transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        cam = GradCAM(model)
        vit = getattr(model, "vit", None)
        atencion = AtencionCLS(vit) if vit is not None and hasattr(vit, "blocks") else None
        del_fold = [r for r in errores if int(r["fold"]) == fold]
        print(f"\n--- fold_{fold}: {len(del_fold)} errores | Grad-CAM en {', '.join(n for n, _ in cam.capas)}")
        for n, r in enumerate(del_fold):
            ruta = os.path.join(args.imagenes, r["ruta"])
            if not os.path.exists(ruta):
                print(f"\n⚠️  No está {ruta}")
                continue
            img = Image.open(ruta).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            i_pred, i_real = clases.index(r["predicha"]), clases.index(r["real"])
            _, m_pred = cam(model, x, i_pred)
            # Con atención CLS, el ViT no lleva Grad-CAM (sale plano en el último bloque)
            # y su panel va una sola vez: no depende de la clase.
            m_pred = sin_vit(m_pred, atencion)
            paneles = [(f"{k} → predicha", v) for k, v in m_pred.items()]
            if atencion is not None:
                paneles.append(("atención CLS (DINOv2)", atencion.mapa((alto, ancho))))
            _, m_real = cam(model, x, i_real)
            paneles += [(f"{k} → real", v) for k, v in sin_vit(m_real, atencion).items()]

            conf = float(r["confianza"])
            sub = "alta_confianza" if conf >= args.umbral_alta else \
                f"{nombre_seguro(r['real'])}__a__{nombre_seguro(r['predicha'])}"
            os.makedirs(os.path.join(args.salida, sub), exist_ok=True)
            archivo = os.path.join(sub, f"{conf:.3f}_{nombre_seguro(os.path.basename(r['ruta']))}")
            guardar(np.asarray(img.resize((ancho, alto))), paneles,
                    f"{r['ruta']}  (fold {fold})\nreal: {r['real']} | predicha: {r['predicha']} ({conf:.1%})",
                    os.path.join(args.salida, archivo))
            filas.append({**{k: r[k] for k in ("fold", "ruta", "real", "predicha", "confianza")},
                          "png": archivo})
            print(f"\r  {n + 1}/{len(del_fold)}", end="")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(os.path.join(args.salida, "errores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "ruta", "real", "predicha", "confianza", "png"])
        w.writeheader()
        w.writerows(filas)
    print(f"\n✅ {len(filas)} Grad-CAM en {os.path.abspath(args.salida)}")


if __name__ == "__main__":
    main()
