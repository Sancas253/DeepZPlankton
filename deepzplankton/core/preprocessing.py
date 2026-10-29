"""Preprocessing registry.

Cada entrada: callable(np.ndarray float32 [B,H,W,3]) -> np.ndarray listo para
el grafo ONNX. Todo numpy puro: la inferencia no importa torch ni tensorflow.

El layout de salida NO es uniforme y eso es a propósito:

 - Los modelos exportados con torch.onnx reciben BCHW y esperan la
   normalización mean/std de ImageNet hecha acá afuera.
 - Los exportados desde Keras con tf2onnx reciben BHWC y llevan la
   normalización DENTRO del grafo. En
   efficientnetb3_finetuned_ensemble/fold_0.onnx los primeros nodos son
       Mul(1/255) → Add(-[0.485,0.456,0.406]) → Mul(1/[0.229,0.224,0.225])
   sobre una entrada [N,300,300,3]: esperan la imagen CRUDA en [0,255].
   Volver a escalar acá (p. ej. x/127.5-1) deja al modelo viendo una imagen
   casi constante, ~0.85% del rango con el que fue entrenado.
"""
import numpy as np


def _preproc_imagenet_meanstd(batch):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    batch = (batch / 255.0 - mean) / std
    return batch.transpose(0, 3, 1, 2).astype(np.float32)  # BHWC -> BCHW


PREPROCESSORS = {
    "imagenet_meanstd": _preproc_imagenet_meanstd,                                         # BCHW
    "rescale_0_1":      lambda x: (x / 255.0).transpose(0, 3, 1, 2).astype(np.float32),   # BCHW
    "identity":         lambda x: x.transpose(0, 3, 1, 2).astype(np.float32),             # BCHW
    "efficientnet":     lambda x: x.astype(np.float32),                                   # BHWC crudo [0,255]
    "convnext":         lambda x: x.astype(np.float32),                                   # BHWC crudo [0,255]
}

# Layout que produce cada preprocesado. El asistente de meta.json lo compara
# contra la entrada real del grafo (onnx_probe.describe) para no ofrecer una
# combinación que después falla o, peor, clasifica mal en silencio.
PREPROC_LAYOUT = {
    "imagenet_meanstd": "NCHW",
    "rescale_0_1":      "NCHW",
    "identity":         "NCHW",
    "efficientnet":     "NHWC",
    "convnext":         "NHWC",
}
