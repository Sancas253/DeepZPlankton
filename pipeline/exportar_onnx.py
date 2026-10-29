"""
exportar_onnx.py — export ONNX de una carpeta de entrenamiento ya descomprimida.

build_model / export_onnx / export_onnx_tf son copia textual de
../average_and_export_onnx.py; export_folds() es la parte de process_zip()
que viene después de extraer el zip. Mismo resultado: fold_N.onnx + meta.json
(framework=onnx, n_folds, best_fold) + metrics.json, el layout que carga
DeepZPlankton V3.

EfficientNetB3 (TF) necesita tf2onnx. convnext_tiny (TF) no se puede exportar
(ver el docstring de average_and_export_onnx.py), por eso no está en el pipeline.
"""
import os
import json
import shutil
import tempfile
import zipfile

import numpy as np



def build_model(arch, num_classes, image_size=None):
    if arch in ("dinov2_convnext_tiny", "dinov2_swin_tiny"):
        from modelo_dinov2_spatial import DINOv2Spatial
        return DINOv2Spatial(num_classes, backbone=arch.removeprefix("dinov2_"),
                             image_size=image_size[0] if image_size else 224,
                             pretrained=False, transfer=False)
    if arch in ("dinov2_convnext_tiny_transfer", "dinov2_swin_tiny_transfer"):
        from modelo_dinov2_spatial import DINOv2Spatial
        return DINOv2Spatial(num_classes, backbone=arch.removeprefix("dinov2_").removesuffix("_transfer"),
                             image_size=image_size[0] if image_size else 224,
                             pretrained=False)
    if arch == "dinov2_convnext_tiny_plus":
        from modelo_dinov2_spatial import DINOv2ConvNeXtPlus
        return DINOv2ConvNeXtPlus(num_classes, image_size=image_size[0] if image_size else 280,
                                  pretrained=False)
    if arch == "dinov2_ft_convnext_tiny":
        from modelo_dinov2_spatial import DINOv2FTConvNeXt
        return DINOv2FTConvNeXt(num_classes, image_size=image_size[0] if image_size else 224,
                                pretrained=False)
    if arch == "convnext_dinov2_fusion_plus":
        from modelo_dinov2_spatial import ConvNeXtDINOv2FusionPlus
        return ConvNeXtDINOv2FusionPlus(num_classes, image_size=image_size[0] if image_size else 288,
                                        pretrained=False)
    if arch == "swin_tiny":
        import timm
        return timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes
        )
    elif arch == "convnext_tiny":
        import timm
        return timm.create_model(
            "convnext_tiny", pretrained=False, num_classes=num_classes
        )
    elif arch == "convnextv2_tiny":
        import timm
        return timm.create_model(
            "convnextv2_tiny", pretrained=False, num_classes=num_classes
        )
    elif arch == "vit_base_patch14_dinov2":
        import timm
        # El positional embedding se guardó al tamaño de entrenamiento
        return timm.create_model(
            "vit_base_patch14_dinov2", pretrained=False, num_classes=num_classes,
            img_size=image_size[0]
        )
    elif arch == "dinov2_mlp":
        import torch.nn as nn
        from transformers import Dinov2Model, Dinov2Config

        class DINOv2MLP(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                config = Dinov2Config.from_pretrained("facebook/dinov2-base")
                self.backbone = Dinov2Model(config)
                hidden = config.hidden_size
                self.head = nn.Sequential(
                    nn.Linear(hidden, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes),
                )
            def forward(self, x):
                # return_dict=False devuelve tupla — necesario para ONNX tracing.
                out = self.backbone(pixel_values=x, return_dict=False)
                cls = out[0][:, 0]
                return self.head(cls)

        return DINOv2MLP(num_classes)

    elif arch == "resnet50_mlp":
        import torch.nn as nn
        from torchvision import models

        class ResNet50MLP(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.backbone = models.resnet50(weights=None)
                self.backbone.fc = nn.Identity()
                self.head = nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes),
                )
            def forward(self, x):
                return self.head(self.backbone(x))

        return ResNet50MLP(num_classes)

    elif arch == "convnext_dinov2_fusion":
        import torch
        import torch.nn as nn
        import timm
        from transformers import Dinov2Model, Dinov2Config

        class FusionModel(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.convnext = timm.create_model(
                    "convnext_tiny", pretrained=False, num_classes=0, global_pool="avg"
                )
                config = Dinov2Config.from_pretrained("facebook/dinov2-base")
                self.dinov2 = Dinov2Model(config)
                convnext_dim = self.convnext.num_features
                dino_dim = config.hidden_size
                self.head = nn.Sequential(
                    nn.Linear(convnext_dim + dino_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes),
                )
            def forward(self, x):
                c = self.convnext(x)
                # return_dict=False devuelve tupla — necesario para ONNX tracing.
                d = self.dinov2(pixel_values=x, return_dict=False)[0][:, 0]
                return self.head(torch.cat([c, d], dim=1))

        return FusionModel(num_classes)

    else:
        raise ValueError(f"Arquitectura desconocida: {arch}")


def export_onnx_tf(fold_dir, image_size, output_path):
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "tf2onnx.convert",
         "--saved-model", fold_dir,
         "--output", output_path,
         "--opset", "17"],
        capture_output=True, text=True,
        env={**os.environ, "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"},
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])


def export_onnx(model, image_size, output_path):
    import torch
    model.eval()
    h, w = image_size
    dummy = torch.zeros(1, 3, h, w)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=17,
            dynamo=False,
        )




def export_folds(root, output_dir):
    """Exporta todos los fold_k/ de `root` a `output_dir`. Lanza excepción si falla."""
    pt_folds = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and os.path.exists(os.path.join(root, d, "model.pt"))
    )
    tf_folds = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and os.path.exists(os.path.join(root, d, "saved_model.pb"))
    )
    folds = pt_folds or tf_folds
    is_tf = bool(tf_folds) and not pt_folds
    if not folds:
        raise FileNotFoundError(f"No hay folds con model.pt o saved_model.pb en {root}")
    print(f"  Folds encontrados: {folds}  ({'TF SavedModel' if is_tf else 'PyTorch'})")

    with open(os.path.join(root, folds[0], "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    arch        = meta["arch"]
    num_classes = len(meta["class_names"])
    image_size  = tuple(meta["image_size"])
    print(f"  Arquitectura: {arch} | Clases: {num_classes} | Imagen: {image_size}")

    os.makedirs(output_dir, exist_ok=True)
    exported = []
    for i, fold in enumerate(folds):
        print(f"  [{i+1}/{len(folds)}] Exportando {fold}...")
        onnx_path = os.path.join(output_dir, f"fold_{i}.onnx")
        if is_tf:
            export_onnx_tf(os.path.join(root, fold), image_size, onnx_path)
        else:
            import torch
            model = build_model(arch, num_classes, image_size)
            model.load_state_dict(torch.load(os.path.join(root, fold, "model.pt"),
                                             map_location="cpu", weights_only=True))
            model.eval()
            export_onnx(model, image_size, onnx_path)
            del model
        print(f"         → {os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB")
        exported.append(onnx_path)

    # Verificación con onnxruntime (dummy con la forma que declara cada grafo:
    # torch exporta NCHW, tf2onnx NHWC).
    import onnxruntime as ort
    sessions = [ort.InferenceSession(p, providers=["CPUExecutionProvider"]) for p in exported]
    all_probs = []
    for sess in sessions:
        spec = sess.get_inputs()[0]
        shape = [d if isinstance(d, int) else 2 for d in spec.shape]
        logits = sess.run(None, {spec.name: np.random.randn(*shape).astype(np.float32)})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        all_probs.append(e / e.sum(axis=1, keepdims=True))
    avg_probs = np.mean(all_probs, axis=0)
    layout = "NHWC" if sessions[0].get_inputs()[0].shape[-1] == 3 else "NCHW"
    n_out = sessions[0].get_outputs()[0].shape[-1]
    if isinstance(n_out, int) and n_out != num_classes:
        raise RuntimeError(f"el grafo produce {n_out} salidas pero meta.json declara "
                           f"{num_classes} clases")
    print(f"  ✅ Verificación ensemble OK — entrada {layout}, {n_out} clases, "
          f"max_conf promedio: {avg_probs.max(axis=1).mean():.4f}")

    fold_val_accs = []
    for i, fold in enumerate(folds):
        with open(os.path.join(root, fold, "meta.json"), "r", encoding="utf-8") as f:
            fold_val_accs.append((i, json.load(f).get("val_acc", 0.0)))
    best_fold_idx, best_val_acc = max(fold_val_accs, key=lambda x: x[1])
    print(f"  ⭐ Mejor fold: fold_{best_fold_idx} (val_acc={best_val_acc:.4f})")

    out_meta = dict(meta)
    out_meta["framework"] = "onnx"
    out_meta["n_folds"]   = len(folds)
    out_meta["best_fold"] = best_fold_idx
    out_meta.pop("fold", None)
    out_meta["notes"] = (
        out_meta.get("notes", "") +
        f" | Ensemble de {len(folds)} folds (fold_N.onnx). Mejor fold: {best_fold_idx}."
    ).strip(" |")

    # La matriz y el reporte del fold 0 van a metrics.json: el clasificador
    # reparsea meta.json en cada corrida y no los usa.
    metricas = {k: out_meta.pop(k) for k in
                ("confusion_matrix", "classification_report") if k in out_meta}
    if metricas:
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metricas, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(out_meta, f, ensure_ascii=False, indent=2)
    return out_meta
