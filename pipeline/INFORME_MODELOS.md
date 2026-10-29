# Informe de modelos — Pipeline_DeepZ.ipynb

Este informe explica **qué hace cada modelo** del notebook `Pipeline_DeepZ.ipynb` y **por qué se tomó cada decisión**. Cubre el pipeline completo: imágenes con barra de escala → dataset limpio → entrenamiento K-Fold → matriz de confusión → ONNX para DeepZPlankton V3.

El notebook es un "envoltorio". Dentro trae embebido (en base64) el código de `deepz_pipeline.py`, `eliminar_barra.py`, `aumentacion.py`, `exportar_onnx.py` y los scripts `entrenamiento/train_*.py`. Se elige el modelo con el parámetro `MODELO`. Todo lo que sigue sale de leer esos archivos.

---

## Índice

1. [Panorama: los 10 modelos de un vistazo](#1-panorama)
2. [Decisiones comunes a todos los modelos](#2-decisiones-comunes)
   - 2.1 Borrado de la barra de escala
   - 2.2 Filtro de imágenes gigantes y mapeo de clases
   - 2.3 Transfer learning (pesos preentrenados)
   - 2.4 Esquema K-Fold con test/val rotativos
   - 2.5 Augmentación compartida
   - 2.6 Resize sin conservar aspecto y normalización
   - 2.7 Pesos por clase (desbalance)
   - 2.8 Selección del mejor modelo, early stopping y scheduler
   - 2.9 Métricas: accuracy, F1 macro, balanced accuracy y OOF
   - 2.10 Semillas
   - 2.11 Export a ONNX y ensemble de folds
3. [Las tres "familias" de entrenamiento](#3-familias)
4. [Modelo por modelo](#4-modelo-por-modelo)
5. [Tabla comparativa de hiperparámetros](#5-tabla)
6. [Observaciones / cosas a tener en cuenta](#6-observaciones)
7. [Glosario rápido](#7-glosario)

---

<a id="1-panorama"></a>
## 1. Panorama: los 10 modelos de un vistazo

| `MODELO` | Qué red es | Tipo | ¿Qué se entrena? | Resolución | Framework |
|---|---|---|---|---|---|
| `resnet50_legacy` | ResNet-50 + MLP | CNN clásica | Solo la cabeza | 224 | TF/Keras |
| `resnet50_mlp` | ResNet-50 + MLP | CNN clásica | Solo la cabeza | 224 | PyTorch |
| `efficientnetb3` | EfficientNet-B3 + MLP | CNN eficiente | Cabeza, después las últimas 50 capas | 300 | TF/Keras |
| `swin_tiny` | Swin Transformer Tiny | Transformer jerárquico | Cabeza, después las etapas 2 y 3 | 224 | PyTorch/timm |
| `convnext_tiny` | ConvNeXt-Tiny | CNN "moderna" | Cabeza, después las etapas 2 y 3 | 224 | PyTorch/timm |
| `convnext_tiny_plus` | ConvNeXt-Tiny (22k) | CNN moderna | Toda la red (receta avanzada) | 288 | PyTorch/timm |
| `convnextv2_tiny_plus` | ConvNeXt V2-Tiny (FCMAE) | CNN moderna | Toda la red (receta avanzada) | 288 | PyTorch/timm |
| `dinov2_mlp` | DINOv2 ViT-B/14 + MLP | Vision Transformer autosupervisado | Solo la cabeza | 224 | PyTorch/HF |
| `dinov2_ft` | DINOv2 ViT-B/14 | Vision Transformer autosupervisado | Toda la red (receta avanzada) | 224 | PyTorch/timm |
| `convnext_dinov2_fusion` | ConvNeXt-Tiny + DINOv2 + MLP | Fusión CNN + ViT | Solo la cabeza | 224 | PyTorch |

Los modelos forman una **progresión experimental**. Primero vienen las líneas de base viejas (ResNet legacy, EfficientNet). Después, backbones modernos con fine-tuning parcial (Swin, ConvNeXt). Después, backbones congelados de distinto tipo para comparar la *calidad de las features* (ResNet-MLP, DINOv2-MLP, fusión). Al final, el fine-tuning completo con receta moderna (los `*_plus` y `dinov2_ft`).

---

<a id="2-decisiones-comunes"></a>
## 2. Decisiones comunes a todos los modelos

Estas decisiones se aplican igual (o casi igual) en todos los modelos. Se explican una sola vez acá.

### 2.1 Borrado de la barra de escala (`eliminar_barra.py`)

**Problema.** El software de captura estampa en cada imagen una barra de escala con un texto ("1 mm", "0.5 mm", "0.3 mm"). Si se deja, la red puede aprender un **atajo** (*shortcut learning*). Por ejemplo, si una clase se fotografió casi siempre con la escala de 0.3 mm, la red aprende "texto 0.3 mm → esa clase" en lugar de mirar el organismo. Además, el V3 clasifica segmentos que **no** tienen barra, así que entrenar con barra generaría una diferencia entre entrenamiento e inferencia.

**Cómo se resuelve y por qué así:**

1. **Plantillas por mediana.** El sello tiene siempre la misma geometría al píxel y no se escala con la imagen. Por eso se busca la línea negra (componente conexo muy oscuro, alargado, en el 25 % superior de la imagen). Se agrupan los recortes por tamaño de línea y se calcula la **mediana píxel a píxel** de la "opacidad de tinta". La mediana elimina lo que varía entre imágenes (el bicho, el fondo) y deja solo lo constante (el sello).
2. **Filtro de plantillas falsas.** Al recorrer todas las carpetas aparecen grupos chicos de partes oscuras y alargadas de los organismos que por casualidad miden lo mismo. Se descarta todo grupo con menos del 5 % de los ejemplos del grupo más grande (`FRACCION_MIN_PLANTILLA`). Según el comentario del código, había 59 grupos de 3–12 ejemplos contra 1396–5862 de los sellos reales. Si se usaran, borrarían pedazos del organismo.
3. **Template matching** (`TM_CCOEFF_NORMED`) sobre el 30 % superior. El umbral `SCORE_MIN = 0.7` es empírico: en ANNELIDA_POLYCHAETA las imágenes sin sello dieron ≤ 0.61 y las que lo tienen ≥ 0.78. Si el score es menor, la imagen se copia sin tocar.
4. **Inpainting "consciente de la forma".** Se borran *solo* los píxeles de tinta (más una dilatación de 1 px). Cada píxel borrado se clasifica como "bicho" o "fondo" según cuál le queda más cerca (transformada de distancia). Después se rellena con `cv2.inpaint` (Telea) usando como fuente **solo** píxeles de ese mismo lado. Así el borde entre organismo y fondo no se "licúa", que es lo que pasaría con un inpaint común si la barra toca al organismo.
5. **Salida siempre en PNG.** Hay dos motivos: `tf.io.decode_image` (EfficientNet) no lee TIFF, y así el dataset queda en el mismo formato que los segmentos que produce el V3.

### 2.2 Filtro de imágenes gigantes y mapeo de clases

- **`MAX_MEGAPIXELES = 20`.** Un recorte típico tiene unos 0,02 MP y el más grande del Training set III tiene 4,3 MP. Una imagen de decenas de MP es casi seguro un escaneo completo que quedó por error en una carpeta de clase. No sirve para entrenar y hace muy lenta cada época, así que se excluye y se registra en `imagenes_descartadas.txt`.
- **`MAPEO_JSON`** permite renombrar, unir o descartar carpetas (`null`) sin tocar los datos originales. Sirve para experimentar con taxonomías distintas.
- **El dataset limpio se guarda como `.zip` sin compresión.** El PNG ya está comprimido, así que deflate no ahorra casi nada y tarda mucho. Se reutiliza con `REUSAR_LIMPIO` para no repetir el borrado de barras.

### 2.3 Transfer learning (pesos preentrenados)

Los modelos originales parten de pesos preentrenados (ImageNet-1k, ImageNet-22k
o DINOv2 LVD-142M).

**Por qué:** el dataset de plancton tiene del orden de decenas de miles de imágenes y muchas clases desbalanceadas. Entrenar una red de 28–86 M de parámetros desde cero con eso produciría sobreajuste. Los pesos preentrenados ya saben detectar bordes, texturas y formas. Solo hay que *adaptarlos* al dominio.

La gran diferencia entre modelos es **cuánto** se adapta el backbone:
- **Congelado** (solo se entrena la cabeza): es barato y mide la calidad "cruda" de las features.
- **Fine-tuning parcial** (cabeza primero, después las últimas etapas): es el esquema clásico.
- **Fine-tuning completo con layer decay**: es la receta moderna y la más potente.

### 2.4 Esquema K-Fold con test/val rotativos

```
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold k:  test = bloque k,  val = bloque (k+1) % 5,  train = los otros 3
```

- **Estratificado:** cada bloque mantiene la proporción de clases. Es imprescindible con clases raras, porque si no, un bloque podría quedarse sin ejemplos de alguna.
- **Tres conjuntos separados (train 60 % / val 20 % / test 20 %).** La validación se usa para *tomar decisiones*: early stopping, elegir la mejor época y bajar el LR. Si se usara el mismo conjunto para evaluar, la métrica quedaría inflada (fuga de información). El test **nunca** influye en el entrenamiento, así que su métrica es honesta.
- **Rotativo:** como el test va rotando, **cada imagen es test exactamente una vez**. Eso permite armar las predicciones *out-of-fold* (OOF) sobre el dataset completo (ver 2.9).
- **Mismos folds para todos los modelos** (misma semilla y mismo orden). Así la comparación entre modelos es justa: todos se evalúan sobre exactamente las mismas particiones. Por eso incluso `resnet50_legacy` usa este esquema en vez del split original.
- **Modelo nuevo en cada fold:** se recarga el preentrenado para que ningún fold "herede" lo aprendido en otro.

### 2.5 Augmentación compartida (`aumentacion.py`)

Se aplica **solo al conjunto de entrenamiento**. Validación y test nunca se aumentan, porque se quiere medir el rendimiento sobre imágenes reales.

| Transformación | Valor | Por qué |
|---|---|---|
| Volteo horizontal | sí | La orientación izquierda/derecha no define la especie. |
| Volteo **vertical** | sí | El plancton no tiene "arriba" en el portaobjetos: cualquier orientación es válida. |
| Rotación | ±12° | Variaciones de pose. Es moderada para no deformar demasiado. |
| Traslación | ±6 % | El organismo no siempre está centrado en el recorte. |
| Zoom | 0,92–1,08 | Variación de tamaño aparente. Es chica porque el tamaño real es una pista útil. |
| Brillo / contraste | ±15 % | Diferencias de iluminación y de cámara entre campañas. |
| Blur gaussiano | p = 0,2, σ 0,1–0,7 | Simula imágenes un poco desenfocadas. |
| **Relleno = 255 (blanco)** | — | El fondo de las imágenes es blanco. Rellenar con negro crearía bordes artificiales que la red podría usar como pista. |

**Por qué una sola definición para todos:** si cada modelo tuviera su propia augmentación, no se podría saber si un modelo gana por su arquitectura o por su augmentación. Hay una versión PyTorch (`transforms_entrenamiento`) y una TF (`aumentar_tf`) que reproduce lo mismo con ops de TensorFlow, para que EfficientNet tenga la misma augmentación. La configuración usada queda guardada en `aumentacion.json`.

**Excepción:** `resnet50_legacy` la desactiva siempre, porque replica un entrenamiento viejo que no tenía augmentación.

### 2.6 Resize sin conservar aspecto y normalización

- Todas las imágenes se llevan a un cuadrado (`Resize((S, S))`) **sin conservar la relación de aspecto**. Es simple y coincide con lo que hace el V3 al clasificar. Ver la observación en la sección 6.
- **Normalización:** cada backbone se normaliza como en su preentrenamiento. Si no, las features cambian.
  - PyTorch (ResNet, Swin, ConvNeXt, DINOv2): media y desvío de ImageNet (`[0.485, 0.456, 0.406]`, `[0.229, 0.224, 0.225]`).
  - EfficientNetB3 Keras: su propio `preprocess_input`.
  - ResNet50 Keras legacy: estilo Caffe (RGB→BGR y resta de la media), metido **dentro del grafo** para que el ONNX reciba la imagen cruda [0, 255].

### 2.7 Pesos por clase (desbalance)

```python
cw = compute_class_weight("balanced", classes=..., y=labels_de_train_del_fold)
CrossEntropyLoss(weight=cw)
```

**Por qué:** hay clases muy grandes (p. ej. Calanoida) y otras con pocos ejemplos. Sin pesos, la red minimiza la pérdida prediciendo bien las clases grandes e ignorando las raras. `"balanced"` asigna a cada clase un peso inversamente proporcional a su frecuencia, así que cada clase "pesa" lo mismo en total.

- Se calculan **sobre el train de cada fold**, no sobre el dataset entero, para no filtrar información de val/test.
- Lo usan todos **menos `efficientnetb3`**, que replica el original sin pesos.
- `convnextv2_tiny_plus` usa la **raíz cuadrada** de los pesos (ver su sección).

### 2.8 Selección del mejor modelo, early stopping y scheduler

Esto se aplica a todos los modelos PyTorch:
- **Se guarda el estado de la época con mejor `val_acc`** y al final se restaura. Así el modelo exportado es el de su mejor punto, no el de la última época, que puede estar sobreajustado.
- **Early stopping con paciencia:** si `val_acc` no mejora durante N épocas (5, 7 u 8 según el modelo), se corta. Ahorra GPU y evita el sobreajuste.
- **Scheduler:** en los modelos "clásicos" se usa `ReduceLROnPlateau(factor=0.2, patience=3)` sobre `val_loss`. Cuando la pérdida se estanca, se divide el LR por 5 para afinar. En los `*_plus` y `dinov2_ft` se usa *warmup + coseno* (ver familia C).

Los dos modelos TF (`efficientnetb3`, `resnet50_legacy`) **no** restauran la mejor época, porque replican a los originales (ver sección 6).

### 2.9 Métricas: accuracy, F1 macro, balanced accuracy y OOF

Con datos desbalanceados, la **accuracy** engaña: un modelo que acierta siempre la clase mayoritaria puede tener 80 % de accuracy y ser inútil para las raras. Por eso se reportan además:
- **F1 macro:** promedio del F1 de cada clase. Todas las clases pesan igual, así que castiga fallar en las raras.
- **Balanced accuracy:** promedio del recall por clase.
- **Por fold** (media ± desvío entre los 5 folds): muestra qué tan **estable** es el modelo.
- **OOF agrupado (`pooled_oof`):** se juntan las predicciones de test de los 5 folds. Como cada imagen fue test una vez, se obtiene una matriz de confusión sobre **todo el dataset**, cada imagen predicha por un modelo que nunca la vio. Para las clases raras es mucho más confiable que mirar un fold suelto, que puede tener 3 ejemplos. De ahí salen las matrices `matriz_confusion.png` (normalizada por fila = recall) y `matriz_confusion_conteos.png`.

### 2.10 Semillas

`SEED = 42` para los folds, y `SEED + fold_idx` para inicializar cada modelo. Hace los experimentos **reproducibles** y comparables. Cada fold usa una semilla distinta, así que no son copias idénticas.

### 2.11 Export a ONNX y ensemble de folds (`exportar_onnx.py`)

- **¿Por qué ONNX?** El V3 corre en Windows sin PyTorch ni TensorFlow, solo con `onnxruntime`, que es liviano y portable. Opset 17, batch dinámico.
- **Cada fold se exporta por separado** (`fold_0.onnx … fold_4.onnx`). El V3 hace un **ensemble**: promedia las probabilidades (softmax) de los 5 modelos. Un ensemble casi siempre es mejor y más estable que cualquier modelo individual, porque cada uno se equivoca en imágenes distintas. Además se aprovechan los 5 modelos que el K-Fold ya entrenó.
- `meta.json` guarda `arch`, `image_size`, `class_names`, `preprocessing` y `best_fold`. El V3 lo lee para saber cómo preprocesar y cómo reconstruir la arquitectura.
- Se verifica el ensemble con `onnxruntime` antes de empaquetar, y se controla que la cantidad de salidas coincida con la cantidad de clases.
- **Esta exportabilidad condicionó elecciones de diseño:** el ConvNeXt de Keras se descartó porque `tf2onnx` no traduce `Erfc` (GELU exacto). Por eso se reescribió en PyTorch/timm (ver `convnext_tiny`). El DINOv2 fine-tuneado usa la versión timm en vez de la de HuggingFace porque soporta layer decay y exporta bien.

---

<a id="3-familias"></a>
## 3. Las tres "familias" de entrenamiento

Los 10 modelos usan una de estas tres recetas. Explicarlas una vez evita repetir.

### Familia A — Backbone congelado + cabeza MLP (*linear probing* con MLP)
Modelos: `resnet50_legacy`, `resnet50_mlp`, `dinov2_mlp`, `convnext_dinov2_fusion`.

```
imagen → [BACKBONE congelado] → vector de features → Linear(→512) → ReLU → Dropout → Linear(→N clases)
```

- Solo se entrena la cabeza (unos cientos de miles de parámetros). Adam, LR 1e-3, hasta 30 épocas, paciencia 7.
- **Por qué:** es rápido y no puede "romper" las features preentrenadas. Además, **aísla la calidad del backbone**. Como la cabeza es la misma en ResNet-MLP, DINOv2-MLP y la fusión (`512, ReLU, Dropout 0.3`), las diferencias de rendimiento se deben solo a las features.
- **Dropout:** apaga neuronas al azar durante el entrenamiento para que la cabeza no memorice.
- **Límite:** las features no se adaptan al plancton. Si el backbone no distingue dos especies parecidas, la cabeza tampoco puede hacerlo.

### Familia B — Fine-tuning en dos fases (congelar → descongelar parcial)
Modelos: `swin_tiny`, `convnext_tiny`, `efficientnetb3`.

```
Fase 1: backbone congelado, se entrena solo la cabeza (LR 1e-3, 15 épocas)
Fase 2: se descongelan las últimas etapas + cabeza  (LR 1e-5, 20 épocas)
```

- **Por qué dos fases:** al principio la cabeza es aleatoria y produce gradientes enormes y ruidosos. Si el backbone estuviera descongelado, esos gradientes destruirían las features preentrenadas (*catastrophic forgetting*). Primero se "acomoda" la cabeza y después se ajusta el backbone con suavidad.
- **Por qué solo las últimas etapas:** las primeras capas de una red detectan cosas genéricas (bordes, colores) que sirven para cualquier imagen. Las últimas detectan conceptos específicos de ImageNet (perros, autos). Lo que hay que reaprender para el plancton está en las capas finales.
- **LR 100 veces más bajo en la fase 2 (1e-5):** se busca mover los pesos *un poco*, no reentrenarlos.
- Se crea un optimizador nuevo en la fase 2, porque los momentos de Adam de la fase 1 no aplican a los parámetros nuevos.

### Familia C — Fine-tuning completo con "receta moderna"
Modelos: `convnext_tiny_plus`, `convnextv2_tiny_plus`, `dinov2_ft`.

Es la receta del paper de ConvNeXt / de fine-tuning de ViTs. **Una sola fase, toda la red entrenable**, pero con varios mecanismos que lo hacen estable:

| Técnica | Qué hace | Por qué |
|---|---|---|
| **AdamW** + weight decay 0,05 | Adam con regularización L2 desacoplada | Es el estándar para ConvNeXt/ViT y regulariza mejor que Adam + L2. |
| **Layer-wise LR decay** (0,8 / 0,75) | Cada capa, yendo hacia la entrada, recibe LR × 0,8 respecto de la siguiente | Resuelve lo mismo que las "dos fases", pero de forma continua: las capas genéricas se mueven poco y las finales mucho. |
| **Warmup 3 épocas + coseno** (LR 1e-6 → LR → 1e-6) | Sube el LR de a poco y después lo baja con forma de coseno | El warmup evita que la cabeza aleatoria rompa el backbone en las primeras iteraciones (el problema de la familia B). El coseno termina con LR muy bajo para afinar. |
| **Label smoothing 0,1** | El objetivo es 0,9 en la clase correcta y el 0,1 restante repartido | Evita que el modelo se vuelva sobreconfiado y mejora la calibración. Ayuda cuando hay etiquetas ambiguas entre especies parecidas. |
| **Drop path 0,1** (stochastic depth) | Durante el entrenamiento se saltean bloques residuales al azar | Es un regularizador fuerte para redes profundas: entrena un "ensemble implícito" de subredes. |
| **EMA de pesos** (decay 0,999) | Se mantiene un promedio móvil de los pesos; se valida y se guarda ese | El promedio es más suave y generaliza mejor que los pesos de cualquier iteración puntual. |
| **Precisión mixta (fp16)** | Cálculo en media precisión en GPU | Casi el doble de velocidad y menos memoria, sin pérdida de calidad (con `GradScaler`). |
| **`drop_last=True`** | Descarta el último batch incompleto | Un batch de 1–2 imágenes da gradientes muy ruidosos. |
| **`WeightedSmoothCE`** (pérdida propia) | Combina class weights y label smoothing correctamente | Ver abajo. |

**Detalle importante sobre la pérdida `WeightedSmoothCE`.** En PyTorch, `CrossEntropyLoss(weight=w, label_smoothing=0.1)` pondera el término de suavizado con el peso de **cada** clase. Como las clases raras tienen pesos altos, toda imagen termina empujada hacia las clases raras, y la mayoritaria (Calanoida, peso bajo) deja de aprenderse. La pérdida propia usa, para cada imagen, **solo el peso de su clase real** en ambos términos (NLL y suavizado), y normaliza por la suma de pesos del batch. Es una corrección sutil pero importante.

---

<a id="4-modelo-por-modelo"></a>
## 4. Modelo por modelo

### 4.1 `resnet50_legacy` — ResNet-50 congelado + MLP (TF/Keras, réplica histórica)

**Qué red es.** **ResNet-50** (He et al., 2015) es la CNN "clásica" por excelencia. Tiene 50 capas organizadas en bloques residuales: cada bloque aprende una *corrección* `F(x)` que se suma a la entrada (`x + F(x)`). Esas "conexiones de atajo" permiten entrenar redes profundas sin que el gradiente se desvanezca. Usa BatchNorm y ReLU. La salida del último bloque pasa por un *Global Average Pooling* y da un vector de 2048 valores.

**Para qué está.** Es una **réplica fiel** del mejor ResNet50 de la exploración original (`AugmentResnetTrain.py`). No busca ser bueno: es la **línea de base** para medir cuánto mejoró todo lo demás.

**Decisiones (todas copiadas del original a propósito):**
- Familia A: backbone congelado, cabeza `GAP → Dense 512 ReLU → Dropout 0.5 → Dense N softmax`.
- **Sin augmentación.** El original se llamaba "Augment…" pero no tenía. Se fuerza aunque `AUMENTACION=True`.
- Adam con el LR por defecto (1e-3), **20 épocas fijas, sin callbacks**: no hay early stopping ni scheduler.
- **Se queda con el modelo de la última época**, no con el mejor.
- Class weights `"balanced"`.
- Preprocesado Caffe metido en el grafo (`Lambda`), para que el ONNX reciba píxeles crudos [0, 255] como espera el V3.
- **Único cambio respecto al original:** usa el K-Fold común, para que su `cv_results.json` sea comparable con el resto.

### 4.2 `resnet50_mlp` — ResNet-50 congelado + MLP (PyTorch, versión mejorada)

**Qué red es.** La misma ResNet-50 (pesos `IMAGENET1K_V2` de torchvision, una versión de mejor calidad que la de Keras). Se le quita la capa `fc` final y queda el vector de 2048 del GAP.

**Decisiones:**
- Familia A con la **misma cabeza que DINOv2-MLP** (`2048 → 512 → ReLU → Dropout 0.3 → N`). Así la comparación ResNet vs DINOv2 aísla el backbone.
- Augmentación compartida, class weights, mejor época por `val_acc`, early stopping (paciencia 7), ReduceLROnPlateau.
- **BatchNorm forzado a modo `eval`.** Se sobreescribe `train()` para que el backbone quede siempre en eval. Aunque los pesos estén congelados, en modo train las capas BatchNorm **actualizan sus medias y varianzas** con cada batch, así que el backbone "congelado" cambiaría igual. Es el equivalente de `trainable=False` en Keras.
- **Comparado con `resnet50_legacy`**, mide cuánto aportan la augmentación, el early stopping con la mejor época, el scheduler y los mejores pesos, con la misma arquitectura.

### 4.3 `efficientnetb3` — EfficientNet-B3 con fine-tuning en dos fases (TF/Keras)

**Qué red es.** **EfficientNet** (Tan & Le, 2019) es una familia de CNNs diseñada para ser eficiente. Su idea central es el **compound scaling**: en vez de agrandar la red solo en profundidad, solo en ancho o solo en resolución, escala las tres a la vez con proporciones fijas. B3 es una variante mediana (unos 12 M de parámetros) pensada para **300×300**. Sus bloques son **MBConv** (convoluciones *depthwise separable* invertidas, de MobileNetV2) con **Squeeze-and-Excitation**, un mecanismo que aprende a darle más peso a los canales importantes. Activación Swish/SiLU.

**Decisiones:**
- Familia B: fase 1 con el backbone congelado (Adam por defecto, 15 épocas). Fase 2 con **las últimas 50 capas** descongeladas (Adam 1e-5, 20 épocas).
- Cabeza `GAP → Dense 512 ReLU → Dropout 0.5 → Dense N softmax`.
- `ReduceLROnPlateau` sobre **`val_accuracy`** (factor 0,5, paciencia 6).
- **300×300**, la resolución nativa de B3.
- Usa la augmentación compartida (versión TF).
- Replica el script original, y por eso **no usa class weights ni restaura la mejor época** (ver sección 6).
- Se exporta como SavedModel y se convierte a ONNX con `tf2onnx`. La entrada queda en NHWC (formato TF).

### 4.4 `swin_tiny` — Swin Transformer Tiny (PyTorch/timm)

**Qué red es.** **Swin Transformer** (Liu et al., 2021) es un *Vision Transformer jerárquico*. Un ViT común parte la imagen en parches y hace atención **global** entre todos ellos, lo que cuesta O(n²) y da una sola escala. Swin, en cambio:
1. Parte la imagen en parches de 4×4 (`patch4`).
2. Calcula la atención solo **dentro de ventanas locales** de 7×7 parches (`window7`), lo que hace el costo lineal.
3. En capas alternas **desplaza las ventanas** (*shifted windows*), para que la información fluya entre ventanas vecinas.
4. Entre etapas **fusiona parches** (*patch merging*): reduce la resolución y duplica los canales, igual que una CNN. Tiene 4 etapas jerárquicas.

Resultado: combina el sesgo local y multiescala de las CNNs con la capacidad de modelado de la atención. Tiny ≈ 28 M de parámetros.

**Decisiones:**
- Pesos ImageNet (`swin_tiny_patch4_window7_224`). La resolución queda fija en **224** porque las ventanas y el *relative position bias* están atados a ese tamaño.
- Familia B: fase 1 solo `head` (Adam 1e-3, 15 épocas). Fase 2 con `layers.2`, `layers.3` y `head` (Adam 1e-5, 20 épocas). Se descongelan las **dos últimas etapas** de las cuatro.
- Paciencia 5, ReduceLROnPlateau, class weights, augmentación compartida.

### 4.5 `convnext_tiny` — ConvNeXt-Tiny (PyTorch/timm)

**Qué red es.** **ConvNeXt** (Liu et al., 2022, "A ConvNet for the 2020s") es una CNN pura que toma una ResNet y la "moderniza" paso a paso con las ideas que hicieron exitosos a los Transformers:
- Un *stem* "patchify" (convolución 4×4 con stride 4), como los parches de un ViT.
- Proporción de etapas 3:3:9:3, como Swin.
- **Convoluciones depthwise 7×7** (kernel grande, como una ventana de atención local).
- Bloque "invertido" (se expande ×4 en el medio, como el MLP de un Transformer).
- **LayerNorm** en vez de BatchNorm, **GELU** en vez de ReLU y menos capas de normalización y activación.

Resultado: rinde igual o mejor que Swin y sigue siendo una CNN, más simple y rápida. Tiny ≈ 28 M de parámetros. Al ser convolucional, **acepta cualquier resolución**.

**Decisiones:**
- **Se reescribió en PyTorch/timm.** La versión Keras no se puede exportar a ONNX: `tf2onnx` no traduce `Erfc` (el GELU exacto) ni el `StatefulPartitionedCall` de la conv depthwise. En PyTorch el GELU sale como `Erf`, que es un op estándar de ONNX.
- Receta **idéntica a `swin_tiny`** (familia B; en fase 2 se descongelan `stages.2` y `stages.3`). Así la comparación Swin vs ConvNeXt es **Transformer vs CNN con todo lo demás igual**.

### 4.6 `convnext_tiny_plus` — ConvNeXt-Tiny 22k + receta moderna

**Qué red es.** **La misma arquitectura** que `convnext_tiny`, con el mismo ONNX y la misma velocidad en el V3. Cambian los pesos y la forma de entrenar.

**Decisiones:**
- **Pesos `convnext_tiny.fb_in22k_ft_in1k`**: preentrenado en ImageNet-**22k** (14 M de imágenes, 22 000 clases) y después ajustado en 1k. Ve muchísima más variedad visual que un modelo solo de 1k, así que sus features son más generales y transfieren mejor a dominios raros como el plancton.
- **288 px** en vez de 224. Más resolución deja ver más detalle fino (setas, apéndices, texturas), que es clave para distinguir especies parecidas. ConvNeXt lo permite, y el V3 lee el tamaño desde `meta.json`.
- **Familia C completa:** AdamW, layer decay 0,8, warmup 3 + coseno (LR 1e-4 → 1e-6), label smoothing 0,1, drop path 0,1, EMA 0,999, fp16, `WeightedSmoothCE`.
- Hasta 30 épocas, paciencia 8 (más larga porque el coseno mejora tarde).
- **Se valida y se guarda el modelo EMA**, no el "crudo".
- Logea también el F1 macro de validación en cada época.

### 4.7 `convnextv2_tiny_plus` — ConvNeXt V2-Tiny (FCMAE) + receta plus

**Qué red es.** **ConvNeXt V2** (Woo et al., 2023) tiene dos novedades respecto de V1:
1. **Preentrenamiento autosupervisado FCMAE** (*Fully Convolutional Masked AutoEncoder*): se tapa una gran parte de la imagen y la red aprende a **reconstruir los parches tapados**. Para lograrlo tiene que entender textura y estructura fina, no solo "qué objeto es". Recién después se ajusta con etiquetas en 22k → 1k.
2. **GRN (Global Response Normalization)** en cada bloque. Fomenta que los canales compitan y se diversifiquen, y evita el "colapso de features" que aparece al entrenar con FCMAE.

Tiene el mismo tamaño y la misma velocidad que ConvNeXt-Tiny.

**Decisiones (dos cambios respecto del plus, cada uno pensado para un error concreto del plus):**
1. **Backbone V2 / FCMAE:** las features de textura y detalle fino deberían ayudar con las confusiones que le quedaban al plus (Evadne↔Podon, Calanoida↔Cyclopoida).
2. **Class weights con raíz cuadrada (`CLASS_WEIGHTS = "sqrt"`).** Con `"balanced"`, el plus **sobre-predecía las clases raras**: 80 Adult se predecían como Furcilia y la precisión de Furcilia era de ~0,56. `sqrt(w)` todavía favorece a las raras, pero mucho menos: una diferencia de ~3,6× entre clases en vez de ~13×. Es un compromiso entre recall de las raras y precisión.
- **Los dos cambios se pueden revertir por separado** con `AJUSTES` (p. ej. `PESOS='"convnext_tiny.fb_in22k_ft_in1k"'` o `CLASS_WEIGHTS='"balanced"'`). Es un diseño de **ablación**: permite medir cuánto aportó cada cambio por separado.
- `meta.json` guarda `arch = convnextv2_tiny` (derivado del nombre de los pesos) para que el export reconstruya la red correcta.

### 4.8 `dinov2_mlp` — DINOv2 ViT-B/14 congelado + MLP

**Qué red es.** **DINOv2** (Meta, 2023) es un **Vision Transformer (ViT-B/14)**: la imagen se corta en parches de 14×14 (224 px → 16×16 = 256 parches). Cada parche se convierte en un vector, se le agrega un *positional embedding* y un **token CLS** especial, y se procesan con 12 capas de **atención global**: cada parche "mira" a todos los demás. El vector final del token CLS resume toda la imagen (768 dimensiones).

Lo especial de DINOv2 es **cómo se entrenó**: con **autoaprendizaje** (*self-supervised*), sin etiquetas, sobre **142 M de imágenes curadas** (LVD-142M). Usa *self-distillation*: una red "alumno" ve recortes de la imagen e intenta producir lo mismo que una red "maestro" (promedio EMA del alumno) que ve otros recortes. Esto se combina con un objetivo de enmascarado tipo iBOT. Así aprende features visuales muy generales, que funcionan bien "tal cual" para tareas nuevas, sin fine-tuning.

**Decisiones:**
- Versión HuggingFace (`facebook/dinov2-base`), backbone completamente congelado.
- Familia A: el **CLS token** pasa a `Linear(768→512) → ReLU → Dropout 0.3 → Linear(→N)`.
- **Por qué congelado:** DINOv2 está pensado para dar buenas features sin ajuste. Es la prueba más directa de "¿qué tan buenas son estas features para el plancton?". Además es barato.
- Misma cabeza que `resnet50_mlp`: la comparación ResNet-congelado vs DINOv2-congelado dice cuánto mejores son las features autosupervisadas modernas.
- Según el comentario de `dinov2_ft`, llegó a **F1 macro 0,917**, lo que motivó probar el fine-tuning.

### 4.9 `dinov2_ft` — DINOv2 ViT-B/14 con fine-tuning completo

**Qué red es.** El **mismo backbone** que `dinov2_mlp`, pero ahora **se entrenan todos sus pesos**.

**Por qué existe (según el código):** congelado llegó a F1 0,917. La conclusión fue que *el límite era estar congelado, no la calidad de las features*. Además, al ser un ViT (atención global) y no una CNN, **se equivoca en imágenes distintas que ConvNeXt**. Aunque no le gane solo a `convnext_tiny_plus`, es el mejor candidato para un **ensemble entre modelos**.

**Decisiones:**
- **Versión timm** (`vit_base_patch14_dinov2.lvd142m`) en vez de HuggingFace, porque timm soporta layer decay de forma nativa y exporta bien a ONNX.
- **Familia C** con ajustes típicos de ViT:
  - **LR 5e-5** (la mitad que ConvNeXt): los transformers son más sensibles y se desestabilizan con LR altos.
  - **Layer decay 0,75** (más agresivo que 0,8): las capas tempranas del ViT se protegen más. Es el valor usual en fine-tuning de ViT/BEiT/MAE.
- **224 px** y `img_size=IMAGE_SIZE`: DINOv2 viene con el positional embedding para 518 px, y timm lo **reinterpola** a 224. El tamaño **tiene que ser múltiplo de 14** (el tamaño de parche).
- Class weights `"balanced"` por defecto. Se puede pasar a `"sqrt"`.

### 4.10 `convnext_dinov2_fusion` — ConvNeXt-Tiny + DINOv2, congelados, con MLP sobre la concatenación

**Qué red es.** Un modelo de **fusión de features** con **dos backbones en paralelo**:
```
             ┌→ ConvNeXt-Tiny (congelado) → avg pool  → 768 ┐
imagen ──────┤                                              ├→ concat (1536) → MLP 512 → N
             └→ DINOv2-base   (congelado) → token CLS → 768 ┘
```

**Por qué:** las dos redes "ven" la imagen de forma distinta. ConvNeXt es una CNN supervisada en ImageNet, con sesgo local y de textura. DINOv2 es un ViT autosupervisado, con atención global y foco en forma y semántica. Si sus features son **complementarias**, concatenarlas le da a la cabeza más información de la que tiene cualquiera de las dos por separado. Es una alternativa barata al ensemble: un solo modelo y una sola pasada de la cabeza.

**Decisiones:**
- Los dos backbones congelados (familia A). Se entrena solo la cabeza `1536 → 512 → ReLU → Dropout 0.3 → N`.
- La misma normalización ImageNet sirve para los dos.
- **`BATCH_SIZE = 16`** (la mitad que el resto): con dos backbones en memoria a la vez, 32 podría no entrar en la GPU de Colab.
- ConvNeXt usa LayerNorm, así que no hace falta el truco de BatchNorm en eval de ResNet.

---

<a id="5-tabla"></a>
## 5. Tabla comparativa de hiperparámetros

| Modelo | Familia | Resolución | Batch | Optimizador | LR | Épocas | Paciencia | Class weights | Augm. | Mejor época | Extras |
|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet50_legacy | A | 224 | 32 | Adam | 1e-3 | 20 fijas | — | balanced | **no** | **no** (última) | Dropout 0,5 |
| resnet50_mlp | A | 224 | 32 | Adam | 1e-3 | 30 | 7 | balanced | sí | sí | BN en eval, Dropout 0,3 |
| dinov2_mlp | A | 224 | 32 | Adam | 1e-3 | 30 | 7 | balanced | sí | sí | Dropout 0,3 |
| convnext_dinov2_fusion | A | 224 | **16** | Adam | 1e-3 | 30 | 7 | balanced | sí | sí | 2 backbones |
| efficientnetb3 | B | **300** | 32 | Adam | 1e-3 → 1e-5 | 15 + 20 | — | **no** | sí | **no** | últimas 50 capas, Dropout 0,5 |
| swin_tiny | B | 224 | 32 | Adam | 1e-3 → 1e-5 | 15 + 20 | 5 | balanced | sí | sí | etapas 2–3 |
| convnext_tiny | B | 224 | 32 | Adam | 1e-3 → 1e-5 | 15 + 20 | 5 | balanced | sí | sí | etapas 2–3 |
| convnext_tiny_plus | C | **288** | 32 | AdamW | 1e-4 coseno | 30 | 8 | balanced | sí | sí (EMA) | LD 0,8, LS, DP, EMA, fp16 |
| convnextv2_tiny_plus | C | **288** | 32 | AdamW | 1e-4 coseno | 30 | 8 | **sqrt** | sí | sí (EMA) | ídem + FCMAE |
| dinov2_ft | C | 224 | 32 | AdamW | **5e-5** coseno | 30 | 8 | balanced | sí | sí (EMA) | LD **0,75** |

(LD = layer decay, LS = label smoothing, DP = drop path)

---

<a id="6-observaciones"></a>
## 6. Observaciones / cosas a tener en cuenta

Son puntos que surgen de leer el código y conviene tener presentes al interpretar resultados o defender la tesis:

1. **`efficientnetb3` no usa class weights ni restaura la mejor época.** Termina con el modelo de la última época de la fase 2, entrenado sin compensar el desbalance. Es coherente con "replicar el original", pero conviene aclararlo al compararlo con los modelos PyTorch: parte de su diferencia puede deberse a eso y no a la arquitectura. Además, el `val_acc` que guarda es el de la última época, y el export elige `best_fold` con ese número.
2. **En la familia B, el scheduler mira `val_loss` y el early stopping mira `val_acc`.** No es un error, pero son dos criterios distintos: el LR puede bajar aunque la accuracy siga mejorando, y viceversa.
3. **En la familia B, la paciencia y la mejor época se reinician en la fase 2.** `run_training` arranca con `best_val_acc = 0`, así que la fase 2 siempre termina guardando *su* mejor época, aunque sea peor que la mejor de la fase 1. En la práctica la fase 2 casi siempre mejora, pero si no lo hace se pierde el modelo de la fase 1.
4. **El resize a cuadrado deforma los organismos alargados** (p. ej. quetognatos o poliquetos con relación de aspecto 5:1 quedan "aplastados"). Es consistente entre entrenamiento e inferencia, así que no genera diferencias, pero una alternativa a explorar sería hacer *padding* a cuadrado con blanco antes del resize.
5. **El `val_acc` se usa para elegir época y también para elegir `best_fold`**, mientras que la métrica de la tesis es el F1 macro. Con desbalance fuerte, la época con mejor accuracy no siempre es la de mejor F1 macro. Los `*_plus` ya logean el F1 de validación; se podría seleccionar por F1.
6. **El ensemble de 5 folds se evalúa en el V3, no en el OOF.** Las métricas de `cv_results.json` son de modelos *individuales* (cada imagen la predice un solo fold). El ensemble final que usa el V3 debería rendir igual o mejor, pero ese número no está medido en el pipeline: no se puede medir sin un conjunto de test externo, porque cada imagen fue train para 3 de los 5 folds.

---

<a id="7-glosario"></a>
## 7. Glosario rápido

- **Backbone:** la parte de la red que extrae features (todo menos la capa final de clasificación).
- **Cabeza (head):** las capas finales que convierten las features en probabilidades por clase.
- **Transfer learning:** reutilizar una red entrenada en otra tarea (p. ej. ImageNet) como punto de partida.
- **Fine-tuning:** seguir entrenando los pesos preentrenados con los datos propios.
- **CNN:** red convolucional. Procesa la imagen con filtros locales que se deslizan sobre ella.
- **Transformer / ViT:** procesa la imagen como una secuencia de parches que se relacionan entre sí mediante *atención*.
- **Atención:** mecanismo en el que cada parche calcula cuánto "mirar" a cada otro parche y combina su información en consecuencia.
- **Autosupervisado:** preentrenamiento sin etiquetas humanas, donde la red se inventa su propia tarea (reconstruir, igualar vistas, etc.).
- **OOF (out-of-fold):** predicciones de cada imagen hechas por el modelo del fold en el que esa imagen era test.
- **Ensemble:** combinar varios modelos promediando sus predicciones.
- **EMA:** *Exponential Moving Average*, un promedio móvil de los pesos a lo largo del entrenamiento.
- **Layer decay:** LR distinto por capa, más chico cuanto más cerca de la entrada.
- **Label smoothing:** suavizar las etiquetas "duras" (1/0) para evitar la sobreconfianza.
- **Drop path:** saltear al azar bloques residuales enteros durante el entrenamiento.
- **ONNX:** formato estándar e independiente del framework para guardar redes y ejecutarlas con `onnxruntime`.
