# DeepZ-Plankton

Clasificador y cuantizador de zooplancton implementado con redes neuronales.

Trabajo Final de Grado – Ingeniería en Computación (UNMdP)
Autor: Castiñeira, Santiago José · Cliente: INIDEP

DeepZ-Plankton procesa imágenes escaneadas de muestras de zooplancton:

1. elimina la barra de escala,
2. segmenta cada espécimen,
3. lo clasifica en una de las clases del Mar Argentino,
4. guarda los recortes en carpetas por especie,
5. genera un reporte con los atributos morfométricos de cada espécimen.

![Diagrama de subsistemas](docs/diagramas/diagrama_subsistemas.png)

## Estructura del repositorio

```
deepzplankton/          App de escritorio (PyQt5 + ONNX Runtime)
  core/                 Clasificador, preprocesamiento, archivos, actualización de resultados
  segmentation/         Segmentación rápida e interactiva, métricas morfométricas
  gui/                  Ventana principal, hilos de trabajo y widgets
  model/                Metadatos del modelo incluido (los .onnx no se versionan)
  main.py               Punto de entrada
pipeline/               Pipeline de entrenamiento
  Pipeline_DeepZ.ipynb  Notebook principal (Colab): barra → dataset → K-Fold → ONNX
  deepz_pipeline.py     Mismo pipeline por línea de comandos
  entrenamiento/        Scripts de entrenamiento de cada modelo
  INFORME_MODELOS.md    Explicación de cada modelo y sus decisiones
entrenamiento_v1/       Primeras versiones de los scripts de entrenamiento
scripts/                Utilidades (promediado/export ONNX, matriz de confusión)
tests/                  Tests de la app y del pipeline
docs/                   Especificación funcional y diagramas
DeepZPlankton_V2.spec   Configuración de PyInstaller
build.bat               Build del ejecutable en Windows
```

## Ejecutar la app

Requiere Python 3.10.

```bash
pip install -r requirements.txt
python -m deepzplankton.main
```

### Modelo

Los pesos `.onnx` no están en el repositorio, porque cada fold supera el límite de 100 MB de GitHub. Hay dos formas de cargarlos:

- copiar los `fold_*.onnx` en `deepzplankton/model/swin_tiny_finetuned_final_ensemble/`, junto al `meta.json`, o
- usar el botón **Agregar modelo** de la app y elegir la carpeta del modelo.

## Generar el ejecutable (Windows)

```bat
build.bat
```

El ejecutable queda en `dist\DeepZPlanktonV3\DeepZPlanktonV3.exe`. Copiá la carpeta del modelo en `dist\DeepZPlanktonV3\model\`.

Cada push a `main` también corre los tests y genera el ejecutable en GitHub Actions. Se descarga desde la pestaña **Actions**, como artefacto `DeepZPlanktonV3-windows`.

## Entrenar modelos

La forma recomendada es abrir `pipeline/Pipeline_DeepZ.ipynb` en Google Colab y elegir el modelo con el parámetro `MODELO`. Para correrlo localmente:

```bash
pip install -r requirements-train.txt
python pipeline/deepz_pipeline.py --modelo swin_tiny --dataset ../dataset
```

Modelos disponibles en `--modelo`: `resnet50_legacy`, `resnet50_mlp`, `efficientnetb3`, `swin_tiny`, `convnext_tiny`, `convnext_tiny_plus`, `dinov2_mlp`, `dinov2_ft`, `dinov2_ft_convnext_tiny`, `convnext_dinov2_fusion` y `convnext_dinov2_fusion_plus`. Las variantes espaciales DINOv2 → ConvNeXt/Swin se corren con `train_dinov2_spatial.py` (ver [`train_dinov2_spatial.md`](train_dinov2_spatial.md)). El detalle de cada uno está en [`pipeline/INFORME_MODELOS.md`](pipeline/INFORME_MODELOS.md).

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests
```

Los tests `test_pipeline_*` y `test_predicciones_oof` necesitan además las dependencias de entrenamiento.

## Documentación

- [Especificación Funcional](docs/Especificacion_Funcional.pdf)
