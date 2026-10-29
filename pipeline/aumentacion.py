"""
aumentacion.py — augmentación predeterminada para TODOS los modelos del pipeline.

Un solo lugar para definirla: los scripts PyTorch (swin_tiny, convnext_tiny,
convnext_dinov2_fusion, dinov2_mlp, resnet50_mlp) usan transforms_entrenamiento() y
EfficientNetB3 (TF) usa aumentar_tf(), que hace lo mismo con ops de TensorFlow.
Solo se aplica a las imágenes de entrenamiento; validación y test nunca se aumentan.

Es la que ya traían los scripts PyTorch de Modelos.ipynb. El fondo de las
imágenes es blanco, por eso lo que queda descubierto al rotar/trasladar se
rellena con 255 y no con negro.

deepz_pipeline.py la apaga con --sin-aumentacion y guarda la configuración
usada en <entrenamiento>/aumentacion.json.
"""
import json
import os

ACTIVA = True

CONFIG = {
    "volteo_horizontal": True,
    "volteo_vertical": True,          # el plancton no tiene "arriba": vale voltear
    "rotacion_grados": 12,            # ± grados
    "traslacion": 0.06,               # fracción del lado, ± en x e y
    "escala": [0.92, 1.08],           # zoom out / zoom in
    "brillo": 0.15,                   # factor en [1-b, 1+b]
    "contraste": 0.15,                # factor en [1-c, 1+c]
    "blur_prob": 0.2,                 # probabilidad de desenfoque gaussiano 3x3
    "blur_sigma": [0.1, 0.7],
    "relleno": 255,                   # fondo blanco
}


def descripcion():
    return dict(CONFIG, activa=ACTIVA)


def guardar_config(carpeta):
    os.makedirs(carpeta, exist_ok=True)
    with open(os.path.join(carpeta, "aumentacion.json"), "w", encoding="utf-8") as f:
        json.dump(descripcion(), f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------- PyTorch ----

def transforms_entrenamiento(image_size, mean, std):
    """Compose de torchvision para el set de entrenamiento."""
    from torchvision import transforms

    c = CONFIG
    pasos = [transforms.Resize((image_size, image_size))]
    if ACTIVA:
        if c["volteo_horizontal"]:
            pasos.append(transforms.RandomHorizontalFlip())
        if c["volteo_vertical"]:
            pasos.append(transforms.RandomVerticalFlip())
        pasos += [
            transforms.RandomAffine(degrees=c["rotacion_grados"],
                                    translate=(c["traslacion"], c["traslacion"]),
                                    scale=tuple(c["escala"]), fill=c["relleno"]),
            transforms.ColorJitter(brightness=c["brillo"], contrast=c["contraste"]),
            transforms.RandomApply([transforms.GaussianBlur(
                kernel_size=3, sigma=tuple(c["blur_sigma"]))], p=c["blur_prob"]),
        ]
    pasos += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(pasos)


# ---------------------------------------------------------- TensorFlow ----

def aumentar_tf(img):
    """Misma augmentación sobre una imagen TF float32 [H, W, 3] en [0, 255].

    Rotación + traslación + escala van en una sola transformación afín
    (ImageProjectiveTransformV3, que permite rellenar con blanco), igual que
    RandomAffine de torchvision.
    """
    import math
    import tensorflow as tf

    if not ACTIVA:
        return img
    c = CONFIG
    if c["volteo_horizontal"]:
        img = tf.image.random_flip_left_right(img)
    if c["volteo_vertical"]:
        img = tf.image.random_flip_up_down(img)

    alto = tf.cast(tf.shape(img)[0], tf.float32)
    ancho = tf.cast(tf.shape(img)[1], tf.float32)
    ang = tf.random.uniform([], -1.0, 1.0) * c["rotacion_grados"] * math.pi / 180.0
    s = tf.random.uniform([], c["escala"][0], c["escala"][1])
    tx = tf.random.uniform([], -1.0, 1.0) * c["traslacion"] * ancho
    ty = tf.random.uniform([], -1.0, 1.0) * c["traslacion"] * alto
    cx, cy = (ancho - 1.0) / 2.0, (alto - 1.0) / 2.0
    # Matriz salida -> entrada (inversa de: escalar y rotar sobre el centro, trasladar).
    a0, a1 = tf.cos(ang) / s, tf.sin(ang) / s
    b0, b1 = -tf.sin(ang) / s, tf.cos(ang) / s
    a2 = cx - a0 * (cx + tx) - a1 * (cy + ty)
    b2 = cy - b0 * (cx + tx) - b1 * (cy + ty)
    matriz = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[tf.newaxis]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis], transforms=matriz,
        output_shape=tf.shape(img)[:2], fill_value=float(c["relleno"]),
        interpolation="BILINEAR", fill_mode="CONSTANT")[0]

    # Brillo y contraste multiplicativos, como ColorJitter.
    img = img * tf.random.uniform([], 1.0 - c["brillo"], 1.0 + c["brillo"])
    gris = tf.reduce_mean(tf.image.rgb_to_grayscale(img))
    img = (img - gris) * tf.random.uniform([], 1.0 - c["contraste"], 1.0 + c["contraste"]) + gris
    img = tf.clip_by_value(img, 0.0, 255.0)

    def desenfocar(x):
        sigma = tf.random.uniform([], c["blur_sigma"][0], c["blur_sigma"][1])
        g = tf.exp(-tf.constant([1.0, 0.0, 1.0]) / (2.0 * sigma ** 2))
        g = g / tf.reduce_sum(g)
        k = tf.tile((g[:, None] * g[None, :])[:, :, None, None], [1, 1, 3, 1])
        x = tf.pad(x[tf.newaxis], [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        return tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], "VALID")[0]

    img = tf.cond(tf.random.uniform([]) < c["blur_prob"], lambda: desenfocar(img), lambda: img)
    return img
