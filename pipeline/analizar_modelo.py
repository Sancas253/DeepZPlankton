"""
analizar_modelo.py — Grad-CAM + UMAP de un modelo entrenado con el pipeline.

Toma un entrenamiento de deepz_pipeline (carpeta con fold_k/model.pt + meta.json)
y una carpeta de imágenes, y genera:
  predicciones.csv         ruta, clase real (si se conoce), predicha y confianza
  embeddings.npz           vector de features de cada imagen (entrada del Linear final)
  umap_por_clase.png       proyección 2D coloreada por clase real (o predicha)
  umap_aciertos.png        lo mismo, marcando aciertos y errores (si hay etiquetas)
  gradcam/<real>/...png    imagen + mapa de calor de dónde miró el modelo
                           (los errores van primero, en gradcam/_errores)

Imágenes: si la carpeta tiene subcarpetas con nombres de clase (como el dataset
limpio), se usan como etiqueta real; si no, se analizan sin etiqueta.

Sirve para todos los modelos PyTorch del pipeline (no para efficientnetb3 ni
resnet50_legacy, que son TF). Los modelos con dos ramas (fusión, DINOv2+ConvNeXt)
dan un Grad-CAM por rama.

Para UMAP: pip install umap-learn  (si no está, se usa t-SNE de sklearn).

Ejemplos:
  python analizar_modelo.py --modelo /content/drive/MyDrive/deepz_resultados/<corrida>/<modelo> \
      --imagenes /content/deepz/dataset_limpio --salida /content/analisis --solo-test
  python analizar_modelo.py --modelo dinov2_ft/fold_2 --imagenes muestras_nuevas --max-cam 30
"""
import argparse
import csv
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from exportar_onnx import build_model  # noqa: E402

EXTENSIONES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Capas candidatas para Grad-CAM (ruta de atributos, nombre de la rama). Se usan
# las que existen en el modelo; la última etapa espacial de cada backbone.
CAPAS_CAM = [
    ("classifier.stages.-1", "ConvNeXt"),
    ("classifier.layers.-1", "Swin"),
    ("vit.blocks.-1", "DINOv2"),
    ("convnext.stages.-1", "ConvNeXt"),
    ("dinov2.encoder.layer.-1", "DINOv2"),
    ("backbone.encoder.layer.-1", "DINOv2"),
    ("backbone.layer4", "ResNet50"),
    ("stages.-1", "ConvNeXt"),
    ("layers.-1", "Swin"),
    ("blocks.-1", "ViT"),
]


# ==============================================================
# Carga del modelo y de las imágenes
# ==============================================================
def cargar_modelo(ruta, fold):
    carpeta = ruta if os.path.exists(os.path.join(ruta, "model.pt")) else os.path.join(ruta, f"fold_{fold}")
    with open(os.path.join(carpeta, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("framework") == "tf":
        raise SystemExit("Solo modelos PyTorch: este es TensorFlow.")
    model = build_model(meta["arch"], len(meta["class_names"]), meta["image_size"])
    model.load_state_dict(torch.load(os.path.join(carpeta, "model.pt"), map_location="cpu",
                                     weights_only=True))
    split = None
    if os.path.exists(os.path.join(carpeta, "split.json")):
        with open(os.path.join(carpeta, "split.json"), encoding="utf-8") as f:
            split = json.load(f)
    return model.eval(), meta, carpeta, split


def listar_imagenes(carpeta, class_names):
    """Devuelve [(ruta, ruta_relativa, indice_clase_real o None)]."""
    items = []
    for raiz, _, archivos in os.walk(carpeta):
        for nombre in sorted(archivos):
            if not nombre.lower().endswith(EXTENSIONES):
                continue
            ruta = os.path.join(raiz, nombre)
            rel = os.path.relpath(ruta, carpeta)
            partes = rel.replace("\\", "/").split("/")
            real = class_names.index(partes[0]) if len(partes) > 1 and partes[0] in class_names else None
            items.append((ruta, rel, real))
    return items


# ==============================================================
# Grad-CAM
# ==============================================================
def resolver(model, ruta):
    obj = model
    for parte in ruta.split("."):
        if parte.lstrip("-").isdigit():
            try:
                obj = obj[int(parte)]
            except (TypeError, IndexError):
                return None
        elif hasattr(obj, parte):
            obj = getattr(obj, parte)
        else:
            return None
    return obj if isinstance(obj, nn.Module) and not isinstance(obj, nn.Identity) else None


def a_mapa(t):
    """Activación de cualquier backbone -> tensor B x C x H x W."""
    if t.dim() == 3:  # tokens de ViT: B x T x C (con CLS / registros adelante)
        lado = int(math.isqrt(t.shape[1]))
        t = t[:, t.shape[1] - lado * lado:]
        return t.transpose(1, 2).reshape(t.shape[0], t.shape[2], lado, lado)
    if t.shape[1] == t.shape[2] and t.shape[3] != t.shape[2]:  # Swin: B x H x W x C
        return t.permute(0, 3, 1, 2)
    return t


class GradCAM:
    def __init__(self, model):
        self.capas, vistas = [], set()
        for ruta, nombre in CAPAS_CAM:
            modulo = resolver(model, ruta)
            if modulo is not None and id(modulo) not in vistas:
                vistas.add(id(modulo))
                self.capas.append((nombre, modulo))
        self.act, self.grad = {}, {}
        for nombre, modulo in self.capas:
            modulo.register_forward_hook(self._hook(nombre))

    def _hook(self, nombre):
        def hook(_, __, salida):
            t = salida[0] if isinstance(salida, (tuple, list)) else salida
            self.act[nombre] = t
            if t.requires_grad:
                t.register_hook(lambda g: self.grad.__setitem__(nombre, g))
        return hook

    def __call__(self, model, x, clase):
        self.act.clear()
        self.grad.clear()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(x)
            logits[0, clase].backward()
        mapas = {}
        for nombre, _ in self.capas:
            if nombre not in self.grad:  # rama congelada con no_grad (sin gradiente)
                continue
            a, g = a_mapa(self.act[nombre].detach()), a_mapa(self.grad[nombre])
            cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
            cam = F.interpolate(cam.float(), size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            cam = cam - cam.min()
            mapas[nombre] = (cam / cam.max()).cpu().numpy() if cam.max() > 0 else cam.cpu().numpy()
        return logits.detach(), mapas


class AtencionCLS:
    """Atención del token CLS a los parches en el último bloque de un ViT de timm."""

    def __init__(self, vit):
        self.vit, self.qkv = vit, None
        attn = vit.blocks[-1].attn
        self.heads = attn.num_heads
        self.q_norm, self.k_norm = getattr(attn, "q_norm", None), getattr(attn, "k_norm", None)
        attn.qkv.register_forward_hook(lambda _, __, salida: setattr(self, "qkv", salida.detach()))

    def mapa(self, tamano):
        B, N, C3 = self.qkv.shape
        qkv = self.qkv.reshape(B, N, 3, self.heads, C3 // 3 // self.heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        att = ((q[:, :, :1] @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5).softmax(-1)  # B,h,1,N
        prefijo = getattr(self.vit, "num_prefix_tokens", 1)
        att = att[0, :, 0, prefijo:].mean(0)
        lado = int(att.numel() ** 0.5)
        att = att[-lado * lado:].reshape(1, 1, lado, lado).float()
        att = F.interpolate(att, size=tamano, mode="bilinear", align_corners=False)[0, 0]
        att = att - att.min()
        return (att / att.max().clamp_min(1e-8)).cpu().numpy()


def atencion_cls(model):
    """AtencionCLS si el modelo tiene un ViT de timm en .vit (DINOv2), si no None."""
    vit = getattr(model, "vit", None)
    return AtencionCLS(vit) if vit is not None and hasattr(vit, "blocks") else None


def sin_vit(mapas, atencion):
    """Saca los Grad-CAM de ramas ViT cuando se muestran con atención CLS."""
    return mapas if atencion is None else {k: v for k, v in mapas.items() if k not in ("DINOv2", "ViT")}


# ==============================================================
# Embeddings (entrada del Linear final)
# ==============================================================
def capturar_embeddings(model):
    ultimo = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    guardado = {}
    ultimo.register_forward_pre_hook(lambda _, entrada: guardado.__setitem__("x", entrada[0].detach()))
    return guardado


def proyectar_2d(X, semilla):
    try:
        import umap
        return umap.UMAP(n_neighbors=min(15, len(X) - 1), min_dist=0.1,
                         random_state=semilla).fit_transform(X), "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        print("⚠️  umap-learn no está instalado (pip install umap-learn): se usa t-SNE.")
        return TSNE(n_components=2, init="pca", perplexity=min(30.0, (len(X) - 1) / 3),
                    random_state=semilla).fit_transform(X), "t-SNE"


# ==============================================================
# Gráficos
# ==============================================================
def graficar_proyeccion(Z, etiquetas, nombres, titulo, ruta, errores=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 11))
    cmap = plt.get_cmap("tab20", max(20, len(nombres)))
    for k, nombre in enumerate(nombres):
        m = etiquetas == k
        if m.any():
            ax.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.6, color=cmap(k % cmap.N), label=f"{nombre} ({m.sum()})")
    if errores is not None and errores.any():
        ax.scatter(Z[errores, 0], Z[errores, 1], s=22, facecolors="none", edgecolors="black",
                   linewidths=0.8, label=f"errores ({errores.sum()})")
    ax.set_title(titulo, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=6, markerscale=3, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def guardar_cam(img, mapas, titulo, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 1 + len(mapas), figsize=(4 * (1 + len(mapas)), 4.4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(img)
    axes[0].set_title("imagen", fontsize=9)
    for ax, (nombre, cam) in zip(axes[1:], mapas.items()):
        ax.imshow(img)
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.set_title(nombre if nombre.startswith("atención") else f"Grad-CAM {nombre}", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(titulo, fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def nombre_seguro(texto):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in texto)


# ==============================================================
# Principal
# ==============================================================
def main(argv=None):
    p = argparse.ArgumentParser(description="Grad-CAM + UMAP de un modelo del pipeline.")
    p.add_argument("--modelo", required=True,
                   help="Carpeta del entrenamiento (con fold_k/) o de un fold (con model.pt)")
    p.add_argument("--imagenes", required=True, help="Carpeta de imágenes (con subcarpetas por clase o plana)")
    p.add_argument("--salida", default="analisis_modelo")
    p.add_argument("--fold", type=int, default=0, help="Fold a usar si --modelo es la carpeta del entrenamiento")
    p.add_argument("--solo-test", action="store_true",
                   help="Usar solo las imágenes de test de ese fold (split.json): las que el modelo no vio")
    p.add_argument("--max-imagenes", type=int, default=0, help="Tope para el UMAP (0 = todas)")
    p.add_argument("--max-cam", type=int, default=60, help="Cantidad de Grad-CAM (primero los errores)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--semilla", type=int, default=42)
    args = p.parse_args(argv)

    random.seed(args.semilla)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta, carpeta_fold, split = cargar_modelo(args.modelo, args.fold)
    model.to(device)
    clases = meta["class_names"]
    alto, ancho = meta["image_size"]
    print(f"🔹 {meta['arch']} | {carpeta_fold} | {len(clases)} clases | {alto}x{ancho} | {device}")

    items = listar_imagenes(args.imagenes, clases)
    if args.solo_test:
        if split is None:
            raise SystemExit("--solo-test necesita split.json en la carpeta del fold.")
        test = {r.replace("\\", "/") for r in split["test"]}
        items = [it for it in items if it[1].replace("\\", "/") in test]
    elif split is not None:
        vistas = {r.replace("\\", "/") for r in split["train"] + split["val"]}
        n_vistas = sum(it[1].replace("\\", "/") in vistas for it in items)
        if n_vistas:
            print(f"⚠️  {n_vistas} imágenes se usaron para entrenar/validar este fold: "
                  f"usá --solo-test para analizar solo las que no vio.")
    if args.max_imagenes and len(items) > args.max_imagenes:
        items = random.sample(items, args.max_imagenes)
    legibles = []
    for it in items:
        try:
            with Image.open(it[0]) as im:
                im.convert("RGB")
            legibles.append(it)
        except OSError as e:
            print(f"⚠️  Se saltea {it[1]}: {e}")
    items = legibles
    if not items:
        raise SystemExit("No se encontraron imágenes.")
    print(f"📊 {len(items)} imágenes ({sum(r is not None for _, _, r in items)} con etiqueta)")

    tf = transforms.Compose([transforms.Resize((alto, ancho)), transforms.ToTensor(),
                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    os.makedirs(args.salida, exist_ok=True)

    # ---- 1) Predicciones + embeddings ----
    guardado = capturar_embeddings(model)
    embeddings, probs = [], []
    with torch.no_grad():
        for i in range(0, len(items), args.batch):
            lote = torch.stack([tf(Image.open(r).convert("RGB")) for r, _, _ in items[i:i + args.batch]])
            logits = model(lote.to(device))
            probs.append(F.softmax(logits.float(), dim=1).cpu())
            embeddings.append(guardado["x"].float().cpu())
            print(f"\r  inferencia {min(i + args.batch, len(items))}/{len(items)}", end="")
    print()
    probs = torch.cat(probs).numpy()
    X = torch.cat(embeddings).numpy()
    pred = probs.argmax(1)
    conf = probs.max(1)
    real = np.array([-1 if r is None else r for _, _, r in items])
    etiquetado = real >= 0
    errores = etiquetado & (pred != real)

    with open(os.path.join(args.salida, "predicciones.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ruta", "real", "predicha", "confianza", "acierto"])
        for (_, rel, r), pr, c in zip(items, pred, conf):
            w.writerow([rel, "" if r is None else clases[r], clases[pr], f"{c:.4f}",
                        "" if r is None else int(pr == r)])
    np.savez_compressed(os.path.join(args.salida, "embeddings.npz"), X=X, pred=pred, real=real,
                        rutas=np.array([rel for _, rel, _ in items]), clases=np.array(clases))
    if etiquetado.any():
        print(f"✅ Accuracy sobre las etiquetadas: {(pred[etiquetado] == real[etiquetado]).mean():.4f} "
              f"({errores.sum()} errores)")

    # ---- 2) UMAP ----
    if len(items) >= 10:
        Z, metodo = proyectar_2d(X, args.semilla)
        color = np.where(etiquetado, real, pred)
        base = f"{metodo} de las features — {meta['arch']} ({len(items)} imágenes)"
        graficar_proyeccion(Z, color, clases, base + ("\ncolor = clase real" if etiquetado.all()
                                                       else "\ncolor = clase real o predicha"),
                            os.path.join(args.salida, "umap_por_clase.png"))
        if etiquetado.any():
            graficar_proyeccion(Z, color, clases, base + "\ncírculo negro = error",
                                os.path.join(args.salida, "umap_aciertos.png"), errores=errores)
        np.save(os.path.join(args.salida, "proyeccion_2d.npy"), Z)
        print(f"🗺️  {metodo} guardado")

    # ---- 3) Grad-CAM (primero errores, después el resto repartido entre clases) ----
    cam = GradCAM(model)
    atencion = atencion_cls(model)
    print(f"🔥 Grad-CAM sobre: {', '.join(n for n, _ in cam.capas)}")
    orden = [i for i in np.argsort(conf) if errores[i]]
    resto = [i for i in range(len(items)) if not errores[i]]
    random.shuffle(resto)
    resto.sort(key=lambda i: real[i])  # intercala clases al recortar
    por_clase = {}
    for i in resto:
        por_clase.setdefault(real[i] if real[i] >= 0 else pred[i], []).append(i)
    while any(por_clase.values()) and len(orden) < args.max_cam:
        for k in list(por_clase):
            if por_clase[k] and len(orden) < args.max_cam:
                orden.append(por_clase[k].pop())
    for n, i in enumerate(orden[:args.max_cam]):
        ruta, rel, r = items[i]
        img = Image.open(ruta).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        _, mapas = cam(model, x, int(pred[i]))
        mapas = sin_vit(mapas, atencion)
        if atencion is not None:
            mapas["atención CLS (DINOv2)"] = atencion.mapa((alto, ancho))
        if not mapas:
            print("⚠️  Ninguna capa del modelo recibe gradiente: no se puede hacer Grad-CAM.")
            break
        txt_real = "?" if r is None else clases[r]
        sub = "_errores" if errores[i] else nombre_seguro(txt_real)
        os.makedirs(os.path.join(args.salida, "gradcam", sub), exist_ok=True)
        titulo = f"{rel}\nreal: {txt_real} | predicha: {clases[pred[i]]} ({conf[i]:.1%})"
        guardar_cam(np.asarray(img.resize((ancho, alto))), mapas, titulo,
                    os.path.join(args.salida, "gradcam", sub,
                                 f"{n:03d}_{nombre_seguro(os.path.splitext(os.path.basename(rel))[0])}.png"))
        print(f"\r  Grad-CAM {n + 1}/{min(len(orden), args.max_cam)}", end="")
    print(f"\n✅ Resultados en {os.path.abspath(args.salida)}")


if __name__ == "__main__":
    main()
