# DINOv2 → ConvNeXt-Tiny / Swin-Tiny

Abrí `pipeline/Pipeline_DeepZ.ipynb` en Colab y elegí `dinov2_convnext_tiny`
o `dinov2_swin_tiny`. El notebook contiene el pipeline actualizado.

La cadena es estrictamente:

`imagen 224×224 → DINOv2-base congelado → mapa 768×16×16 → adaptador 1×1 (768→96) → bloques ConvNeXt/Swin preentrenados → clases`

DINOv2 genera las características; el segundo modelo las clasifica. ConvNeXt
parte de `convnext_tiny.fb_in22k_ft_in1k`, como `convnext_tiny_plus`, y Swin de
`swin_tiny_patch4_window7_224.ms_in1k`, equivalente al Swin original. Se carga
el checkpoint RGB y luego se reemplaza su proyección de entrada por el
adaptador. Se conservan las normalizaciones y los bloques internos. Swin
adapta sus ventanas a 4 con interpolación de los sesgos relativos mediante
`timm.set_input_size`. La salida sigue siendo el clasificador del Tiny.

Esta transferencia entre características DINOv2 y bloques entrenados con RGB
es una hipótesis experimental: conservar pesos no garantiza que mejore la
precisión. La implementación se apoya en las interfaces oficiales de
[timm Swin](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer.py)
y [timm ConvNeXt](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py).

## Entrenamiento

- 30 épocas máximas, batch 32 y resolución 224.
- Primeras 3 épocas: aprenden solo el adaptador y la cabeza; los bloques
  preentrenados quedan congelados y en modo evaluación.
- Épocas restantes: aprende todo el Tiny; DINOv2 permanece congelado y en eval.
- AdamW: LR base 1e-4, layer decay 0.8; adaptador/cabeza con LR objetivo 1e-3.
- Warmup 3 épocas y coseno; weight decay 0.05, drop path 0.1, EMA 0.999.
- WeightedSmoothCE con pesos balanced calculados solo en train y smoothing 0.1,
  igual a los modelos plus. Selección de checkpoint EMA por val_acc, paciencia 8.

Las 3 épocas iniciales están incluidas en las 30. Los ritmos efectivos varían
con warmup, coseno y profundidad. La receta toma como referencia los modelos
plus; no es idéntica a la receta anterior de fusión congelada.

## Comparación justa

Se conserva el mismo ImageFolder, orden de clases, dataset limpio,
StratifiedKFold de 5 particiones con semilla 42, test=k, val=(k+1)%5 y train=resto.
Son 60/20/20, con cada imagen evaluada como test una sola vez. Usa exactamente
`aumentacion.py`, el interruptor AUMENTACION, normalización ImageNet y val_acc
para seleccionar la época. El test se evalúa después de recuperar el checkpoint.
No añade imágenes de zooplancton externas, aumentación especial ni TTA.
El preentrenamiento externo y el coste de cómputo deben declararse como en
los otros modelos: un protocolo común no implica arquitecturas de igual coste.

Cada fold guarda `split.json` con sus rutas relativas para auditar las
particiones. Para comparar, usar el mismo dataset limpio y los mismos valores
de N_SPLITS, SEED y AUMENTACION. No ajustar hiperparámetros mirando los tests/OOF;
una búsqueda sistemática necesitaría validación interna adicional para todos.
No se modifican las recetas de los diez modelos originales.

## Ejecución y salidas

Desde `tesis`:

```powershell
python pipeline/deepz_pipeline.py --modelo dinov2_convnext_tiny --dataset "../dataset" --trabajo "../deepz_dino_convnext"
python pipeline/deepz_pipeline.py --modelo dinov2_swin_tiny --dataset "../dataset" --trabajo "../deepz_dino_swin"
```

En Colab se usa el formulario habitual. Para poca VRAM: `AJUSTES="BATCH_SIZE=8"`.
Para una prueba breve: `N_SPLITS=3 EPOCHS=2 ADAPTER_EPOCHS=1 WARMUP_EPOCHS=1`.
Si se compara con otros modelos, registrar también estos ajustes.
`train_dinov2_spatial.py` es una entrada al mismo pipeline.

Los pesos completos, metadatos, matrices, métricas OOF y ensemble ZIP ONNX
siguen el formato del pipeline. Los metadatos nuevos identifican la arquitectura
con sufijo `_transfer`, distinguiéndola de la primera variante desde cero;
el exportador conserva soporte para los checkpoints anteriores.

```powershell
python -m unittest discover -s tests -p test_pipeline_dinov2_spatial.py -v
```

Las pruebas verifican gradientes y congelamiento por fase, grupos del optimizador,
particiones, checkpoint, matrices, ZIP, concordancia PyTorch/ONNX y contenido
embebido del notebook. No sustituyen una evaluación real de precisión.

## Variante `dinov2_convnext_tiny_plus`

`imagen 280×280 → DINOv2-base congelado (últimas 4 capas, mezcla aprendida) → mapa 768×20×20 → LayerNorm + 1×1 (768→384) → etapas 2-3 de ConvNeXt-Tiny → [pooling ConvNeXt ‖ CLS DINOv2] → clases`

Diferencias con `dinov2_convnext_tiny`:

- **Escala coherente.** Los patches de DINOv2 tienen stride 14; la etapa 2 de
  ConvNeXt trabaja a stride 16 con 384 canales. Las features entran ahí en vez
  de reemplazar el stem (stride 4), que obligaba a comprimir 768→96 y reducir
  hasta 2×2. Se descartan stem y etapas 0-1; quedan las etapas 2-3 (la mayor
  parte de los pesos preentrenados).
- **Últimas 4 capas** de DINOv2 normalizadas y mezcladas con pesos softmax
  aprendidos, como en la evaluación lineal de DINOv2.
- **Token CLS** (descriptor global) concatenado al pooling de ConvNeXt, con dropout 0.2.
- **280 px** (grilla 20×20; debe ser múltiplo de 28), en línea con los 288 de los plus.
- **Layer decay 0.9** (en vez de 0.8): los primeros bloques de la etapa 2
  reciben features nuevas y necesitan adaptarse más.

El resto (folds, aumentación, loss, EMA, selección por val_acc, ONNX) es igual.
Es una hipótesis a validar contra `dinov2_convnext_tiny`, `dinov2_mlp` y
`convnext_tiny_plus` con el mismo protocolo.

```powershell
python pipeline/deepz_pipeline.py --modelo dinov2_convnext_tiny_plus --dataset "../dataset" --trabajo "../deepz_dino_convnext_plus"
python -m unittest discover -s tests -p test_pipeline_dinov2_convnext_plus.py -v
```

## Variante `dinov2_ft_convnext_tiny`

Igual que `dinov2_convnext_tiny_plus`, pero DINOv2 **se entrena** (ViT de timm
`vit_base_patch14_dinov2.lvd142m`, el mismo de `dinov2_ft`) a 224 px.

- Primeras 3 épocas: solo adaptador, mezcla de capas y cabeza (ViT y ConvNeXt
  congelados, para que el adaptador aleatorio no dañe las features de DINOv2).
- Después: todo junto. ViT con LR 5e-5 y layer decay 0.75 (receta `dinov2_ft`);
  ConvNeXt con LR 1e-4 y layer decay 0.9; adaptador/cabeza con 1e-3.
- Cuesta un poco más que `dinov2_ft` (las etapas 2-3 de ConvNeXt son ~25M
  parámetros extra). Si falta VRAM: `AJUSTES="BATCH_SIZE=16"`.

Compararlo contra `dinov2_ft`: si no lo supera, ConvNeXt no aporta en serie y
conviene un ensemble `dinov2_ft` + `convnext_tiny_plus`.

```powershell
python pipeline/deepz_pipeline.py --modelo dinov2_ft_convnext_tiny --dataset "../dataset" --trabajo "../deepz_dino_ft_convnext"
python -m unittest discover -s tests -p test_pipeline_dinov2_ft_convnext.py -v
```
