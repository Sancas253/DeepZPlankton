"""
average_and_export_onnx.py

Para cada zip de modelos pasado por argumento (o todos los .zip del Escritorio
si no se pasa ninguno):
  1. Extrae los folds
  2. Exporta cada fold individualmente como fold_N.onnx (ensemble — NO weight averaging)
  3. Guarda el resultado en Escritorio/onnx/<nombre>/

En inferencia, classifierExe.py detecta los fold_N.onnx y promedia los softmax (ensemble).

Uso:
  python average_and_export_onnx.py                              # todos los .zip del Escritorio
  python average_and_export_onnx.py modelo.zip otro_modelo.zip    # solo esos
  python average_and_export_onnx.py C:\ruta\a\modelo.zip          # ruta completa

Límite conocido — SavedModel de TF con capas custom (probado 2026-08-21):
  convnext_tiny_finetuned_final.zip no se puede convertir. tf2onnx no traduce
  el op Erfc (GELU exacto de esa implementación de ConvNeXt-Tiny en Keras) ni
  logra inlinear el StatefulPartitionedCall en el que queda envuelta la
  depthwise conv. Probado con TF 2.21, TF 2.15 y con freezing manual del grafo
  (convert_variables_to_constants_v2 + tf2onnx.convert.from_graph_def): mismo
  resultado en los tres casos. No es un problema de entorno (numpy/protobuf):
  es que tf2onnx no soporta esos ops para esta arquitectura tal como está
  guardada. Para convertirlo haría falta reescribir el modelo (portar los
  pesos a un ConvNeXt-Tiny de timm en PyTorch y exportar con torch.onnx, como
  los demás modelos) o parchear la implementación Keras para usar ops
  soportados (GELU aproximado en vez de exacto, depthwise conv sin wrapper).
"""
import os
import json
import shutil
import tempfile
import zipfile

import numpy as np

DESKTOP = r"C:\Users\santi\Escritorio"


def build_model(arch, num_classes):
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


def process_zip(name, zip_path):
    if not os.path.exists(zip_path):
        print(f"  [SKIP] No encontrado: {zip_path}")
        return

    output_dir = os.path.join(DESKTOP, "onnx", name)
    if os.path.exists(output_dir):
        print(f"  [SKIP] Ya existe: {output_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Procesando: {name}")
    print(f"{'='*60}")

    tmpdir = tempfile.mkdtemp(prefix="deepz_ens_")
    try:
        print("  Extrayendo zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        # Si el zip tiene una única entrada en el nivel superior y es un
        # directorio, normalmente es una carpeta contenedora (ej. zip de
        # todo el OUTPUT_DIR con varios folds adentro) y hay que bajar un
        # nivel. Pero con un solo fold sin contenedor esa única entrada es
        # el fold en sí (tiene model.pt/saved_model.pb directo adentro):
        # ahí NO hay que bajar, porque un nivel más abajo no hay nada.
        entries = [e for e in os.listdir(tmpdir) if not e.startswith("__")]
        single_entry_is_fold_itself = (
            len(entries) == 1 and os.path.isdir(os.path.join(tmpdir, entries[0]))
            and (os.path.exists(os.path.join(tmpdir, entries[0], "model.pt"))
                 or os.path.exists(os.path.join(tmpdir, entries[0], "saved_model.pb")))
        )
        root = tmpdir if single_entry_is_fold_itself else (
            os.path.join(tmpdir, entries[0]) if (
                len(entries) == 1 and os.path.isdir(os.path.join(tmpdir, entries[0]))
            ) else tmpdir
        )

        # Detectar folds PT o TF
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
        folds    = pt_folds or tf_folds
        is_tf    = bool(tf_folds) and not pt_folds

        if not folds:
            print("  [ERROR] No se encontraron folds con model.pt o saved_model.pb")
            return
        print(f"  Folds encontrados: {folds}  ({'TF SavedModel' if is_tf else 'PyTorch'})")

        meta_path = os.path.join(root, folds[0], "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        arch        = meta["arch"]
        num_classes = len(meta["class_names"])
        image_size  = tuple(meta["image_size"])
        print(f"  Arquitectura: {arch} | Clases: {num_classes} | Imagen: {image_size}")

        os.makedirs(output_dir, exist_ok=True)

        # Exportar cada fold como fold_N.onnx independiente
        exported = []
        for i, fold in enumerate(folds):
            print(f"  [{i+1}/{len(folds)}] Exportando {fold}...")
            onnx_path = os.path.join(output_dir, f"fold_{i}.onnx")
            if is_tf:
                fold_dir = os.path.join(root, fold)
                export_onnx_tf(fold_dir, image_size, onnx_path)
            else:
                import torch
                pt_path = os.path.join(root, fold, "model.pt")
                model = build_model(arch, num_classes)
                model.load_state_dict(
                    torch.load(pt_path, map_location="cpu", weights_only=True)
                )
                model.eval()
                export_onnx(model, image_size, onnx_path)
                del model
            size_mb = os.path.getsize(onnx_path) / 1024 / 1024
            print(f"         → {size_mb:.1f} MB")
            exported.append(onnx_path)

        # Verificar ensemble con onnxruntime.
        # El dummy se arma con la forma que declara CADA grafo, no con una fija:
        # los export de torch reciben NCHW pero los de tf2onnx reciben NHWC, y
        # con un dummy NCHW fijo esta verificación fallaba siempre para los TF.
        # Tampoco se traga la excepción: si el modelo exportado no corre, el
        # export no sirve y hay que enterarse acá.
        import onnxruntime as ort
        sessions = [
            ort.InferenceSession(p, providers=["CPUExecutionProvider"])
            for p in exported
        ]
        all_probs = []
        for sess in sessions:
            spec = sess.get_inputs()[0]
            shape = [d if isinstance(d, int) else 2 for d in spec.shape]
            dummy_np = np.random.randn(*shape).astype(np.float32)
            logits = sess.run(None, {spec.name: dummy_np})[0]
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            all_probs.append(e / e.sum(axis=1, keepdims=True))
        avg_probs = np.mean(all_probs, axis=0)
        layout = "NHWC" if sessions[0].get_inputs()[0].shape[-1] == 3 else "NCHW"
        n_out = sessions[0].get_outputs()[0].shape[-1]
        if isinstance(n_out, int) and n_out != num_classes:
            raise RuntimeError(
                f"el grafo produce {n_out} salidas pero meta.json declara "
                f"{num_classes} clases"
            )
        print(f"  ✅ Verificación ensemble OK — entrada {layout}, {n_out} clases, "
              f"max_conf promedio: {avg_probs.max(axis=1).mean():.4f}")

        # Determinar mejor fold por val_acc
        fold_val_accs = []
        for i, fold in enumerate(folds):
            fold_meta_path = os.path.join(root, fold, "meta.json")
            with open(fold_meta_path, "r", encoding="utf-8") as f:
                fm = json.load(f)
            fold_val_accs.append((i, fm.get("val_acc", 0.0)))
        best_fold_idx, best_val_acc = max(fold_val_accs, key=lambda x: x[1])
        print(f"  ⭐ Mejor fold: fold_{best_fold_idx} (val_acc={best_val_acc:.4f})")

        # meta.json con framework=onnx, n_folds y best_fold
        out_meta = dict(meta)
        out_meta["framework"]  = "onnx"
        out_meta["n_folds"]    = len(folds)
        out_meta["best_fold"]  = best_fold_idx
        # "fold" viene del meta de folds[0] y describiría un solo fold: acá el
        # directorio tiene todos, así que dejarlo en 0 es mentira.
        out_meta.pop("fold", None)
        out_meta["notes"] = (
            out_meta.get("notes", "") +
            f" | Ensemble de {len(folds)} folds (fold_N.onnx). Mejor fold: {best_fold_idx}."
        ).strip(" |")

        # La matriz de confusión y el reporte van aparte: el clasificador
        # reparsea meta.json en cada corrida y no los usa para nada.
        metricas = {k: out_meta.pop(k) for k in
                    ("confusion_matrix", "classification_report") if k in out_meta}
        if metricas:
            with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metricas, f, ensure_ascii=False, indent=2)

        with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(out_meta, f, ensure_ascii=False, indent=2)

        print(f"  LISTO: {output_dir}")

    except Exception as e:
        import traceback
        print(f"  [ERROR] {e}")
        traceback.print_exc()
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if args:
        # Cualquier .zip pasado por argumento (nombre suelto -> se busca en
        # el Escritorio, o ruta completa/relativa).
        zip_paths = []
        for arg in args:
            zip_paths.append(arg if os.path.isabs(arg) or os.sep in arg
                              else os.path.join(DESKTOP, arg))
    else:
        # Sin argumentos: todos los .zip del Escritorio.
        zip_paths = [
            os.path.join(DESKTOP, f)
            for f in sorted(os.listdir(DESKTOP))
            if f.lower().endswith(".zip")
        ]

    for zip_path in zip_paths:
        name = os.path.splitext(os.path.basename(zip_path))[0]
        process_zip(name, zip_path)
    print("\n\nTodo terminado.")
