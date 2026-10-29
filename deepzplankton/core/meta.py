"""Carga y validación del manifest meta.json de un modelo."""
import json
import os

from .paths import model_path

# 'arch' quedó fuera: con ONNX la arquitectura vive dentro del grafo y el campo
# no se usa en ninguna parte de la inferencia. Se sigue aceptando si está.
REQUIRED_FIELDS = {"framework", "image_size", "class_names", "preprocessing"}


def load_meta(model_dir):
    model_dir = model_path(model_dir)
    meta_path = os.path.join(model_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"meta.json not found in '{model_dir}'.\n"
            "Each model directory must contain a meta.json manifest."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    missing = REQUIRED_FIELDS - meta.keys()
    if missing:
        raise ValueError(f"meta.json is missing required fields: {missing}")
    return meta
