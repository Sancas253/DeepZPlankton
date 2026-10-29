"""
Lectura liviana de metadatos de un .onnx, sin cargar los pesos.

onnxruntime.InferenceSession() sobre un modelo de 440 MB tarda varios segundos
y reserva memoria para todos los pesos; acá sólo hacen falta la forma de entrada
y la de salida, que viven en ModelProto.graph.{input,output}. Se recorre el
protobuf con mmap salteando los payloads por índice, así que nunca se copia un
initializer.

Sirve para dos cosas en el asistente de meta.json:
  - deducir el layout (NHWC = export de Keras/tf2onnx, NCHW = export de torch),
    que determina qué preprocesado corresponde;
  - leer el nº de salidas del grafo y validar contra class_names, en lugar de
    confiar en que el usuario cuente bien.
"""
import mmap
import os


def _varint(buf, i):
    result = shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7


def _walk(buf, start, end):
    """Itera (nº de campo, wire type, valor) sobre un mensaje protobuf.

    Para wire type 2 el valor es el par (inicio, fin) del payload: se devuelven
    índices, nunca bytes, para no copiar tensores de cientos de MB.
    """
    i = start
    while i < end:
        key, i = _varint(buf, i)
        field, wtype = key >> 3, key & 7
        if wtype == 0:
            value, i = _varint(buf, i)
            yield field, wtype, value
        elif wtype == 2:
            length, i = _varint(buf, i)
            yield field, wtype, (i, i + length)
            i += length
        elif wtype == 5:
            i += 4
        elif wtype == 1:
            i += 8
        else:
            raise ValueError(f"wire type inesperado: {wtype}")


def _dims(buf, span):
    """ValueInfoProto → lista de dimensiones (int, o None si es dinámica)."""
    dims = []
    for f1, w1, type_span in _walk(buf, *span):            # ValueInfoProto.type
        if f1 != 2 or w1 != 2:
            continue
        for f2, w2, tensor_span in _walk(buf, *type_span):  # TypeProto.tensor_type
            if f2 != 1 or w2 != 2:
                continue
            for f3, w3, shape_span in _walk(buf, *tensor_span):  # .shape
                if f3 != 2 or w3 != 2:
                    continue
                for f4, w4, dim_span in _walk(buf, *shape_span):  # .dim
                    if f4 != 1 or w4 != 2:
                        continue
                    value = None
                    for f5, w5, v5 in _walk(buf, *dim_span):
                        if f5 == 1 and w5 == 0:             # dim_value
                            value = v5
                    dims.append(value)
    return dims


def probe(onnx_path):
    """Devuelve {'input_shape': [...], 'output_shape': [...]} del primer IO."""
    with open(onnx_path, "rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf:
            for field, wtype, span in _walk(buf, 0, len(buf)):
                if field != 7 or wtype != 2:                # ModelProto.graph
                    continue
                inputs, outputs = [], []
                for gfield, gwtype, gspan in _walk(buf, *span):
                    if gwtype != 2:
                        continue
                    if gfield == 11:                        # GraphProto.input
                        inputs.append(_dims(buf, gspan))
                    elif gfield == 12:                      # GraphProto.output
                        outputs.append(_dims(buf, gspan))
                return {
                    "input_shape":  inputs[0] if inputs else None,
                    "output_shape": outputs[0] if outputs else None,
                }
    return None


def first_onnx(model_dir):
    """Nombre del .onnx representativo del directorio, o None."""
    try:
        entries = os.listdir(model_dir)
    except OSError:
        return None
    folds = sorted(f for f in entries if f.startswith("fold_") and f.endswith(".onnx"))
    if folds:
        return folds[0]
    return "model.onnx" if "model.onnx" in entries else None


def describe(model_dir):
    """Deduce layout, tamaño de imagen y nº de clases de un directorio ONNX.

    Devuelve None si el directorio no contiene ONNX o no se pudo interpretar,
    de modo que el llamador pueda caer al comportamiento anterior.
    """
    name = first_onnx(model_dir)
    if name is None:
        return None
    try:
        info = probe(os.path.join(model_dir, name))
    except Exception:
        return None
    if not info or not info["input_shape"]:
        return None

    dims = info["input_shape"]
    out = info["output_shape"] or []
    result = {
        "file":       name,
        "n_folds":    len([f for f in os.listdir(model_dir)
                           if f.startswith("fold_") and f.endswith(".onnx")]),
        "n_classes":  out[-1] if out and isinstance(out[-1], int) else None,
        "layout":     None,
        "image_size": None,
    }
    if len(dims) == 4:
        if dims[1] == 3:            # N,C,H,W — export de torch.onnx
            result["layout"] = "NCHW"
            result["image_size"] = [dims[2], dims[3]]
        elif dims[3] == 3:          # N,H,W,C — export de Keras vía tf2onnx
            result["layout"] = "NHWC"
            result["image_size"] = [dims[1], dims[2]]
    return result
