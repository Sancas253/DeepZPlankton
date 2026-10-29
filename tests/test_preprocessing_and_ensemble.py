"""
Tests numéricos de PREPROCESSORS y del promedio de ensemble.

Estas dos piezas son las que determinan si un modelo predice bien o mal, y
antes no tenían ningún test que verificara los números: un cambio accidental
en una constante (el mean/std de ImageNet, por ejemplo) o en la fórmula de
promediado no rompía nada del suite — se hubiera notado recién como "el
modelo predice raro" en producción.
"""
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deepzplankton.core.classifier as classifierExe
from deepzplankton.core.classifier import PREPROCESSORS, PREPROC_LAYOUT


# ============================================================
# Preprocesado: valores exactos y layout declarado
# ============================================================

class TestPreprocessorsValoresExactos:

    def _batch(self, value, h=2, w=2, c=3):
        """Batch BHWC de 1 imagen, un solo valor repetido en todos los píxeles."""
        return np.full((1, h, w, c), value, dtype=np.float32)

    def test_imagenet_meanstd_normaliza_con_las_constantes_correctas(self):
        batch = self._batch(255.0)  # blanco puro
        out = PREPROCESSORS["imagenet_meanstd"](batch)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        esperado = (1.0 - mean) / std  # (255/255 - mean) / std

        assert out.shape == (1, 3, 2, 2)  # BCHW
        for canal in range(3):
            assert np.allclose(out[0, canal], esperado[canal], atol=1e-6)

    def test_rescale_0_1_divide_por_255(self):
        batch = self._batch(255.0)
        out = PREPROCESSORS["rescale_0_1"](batch)
        assert out.shape == (1, 3, 2, 2)
        assert np.allclose(out, 1.0)

    def test_identity_no_cambia_valores_solo_transpone(self):
        batch = self._batch(42.0)
        out = PREPROCESSORS["identity"](batch)
        assert out.shape == (1, 3, 2, 2)
        assert np.allclose(out, 42.0)

    def test_efficientnet_deja_la_imagen_cruda_en_0_255(self):
        """El grafo de efficientnetb3_finetuned_ensemble normaliza ADENTRO;
        volver a escalar acá lo dejaría viendo una imagen casi constante."""
        batch = self._batch(200.0)
        out = PREPROCESSORS["efficientnet"](batch)
        assert out.shape == (1, 2, 2, 3)  # BHWC, sin transponer
        assert np.allclose(out, 200.0)

    def test_convnext_deja_la_imagen_cruda_en_0_255(self):
        batch = self._batch(123.0)
        out = PREPROCESSORS["convnext"](batch)
        assert out.shape == (1, 2, 2, 3)
        assert np.allclose(out, 123.0)


class TestPreprocLayoutConsistente:
    """PREPROC_LAYOUT es lo que el asistente de meta.json usa para no ofrecer
    una combinación preprocesado/modelo que clasifique mal en silencio. Si
    algún preprocesador cambia de layout sin actualizar el diccionario, el
    asistente queda mintiendo."""

    @pytest.mark.parametrize("nombre", list(PREPROCESSORS.keys()))
    def test_el_layout_declarado_coincide_con_la_forma_real(self, nombre):
        batch = np.zeros((1, 5, 7, 3), dtype=np.float32)  # H=5, W=7 para no ser cuadrado
        out = PREPROCESSORS[nombre](batch)
        layout = PREPROC_LAYOUT[nombre]
        if layout == "NCHW":
            assert out.shape == (1, 3, 5, 7), f"{nombre} declarado NCHW pero da {out.shape}"
        elif layout == "NHWC":
            assert out.shape == (1, 5, 7, 3), f"{nombre} declarado NHWC pero da {out.shape}"
        else:
            pytest.fail(f"layout desconocido para {nombre}: {layout!r}")

    def test_todos_los_preprocesadores_tienen_layout_declarado(self):
        assert set(PREPROCESSORS.keys()) == set(PREPROC_LAYOUT.keys())


# ============================================================
# Ensemble: el promedio de softmax entre folds
# ============================================================

class _FakeSession:
    """Sesión ONNX falsa: siempre devuelve los mismos logits, tileados al
    tamaño del batch que le llega."""

    def __init__(self, logits, input_name="input"):
        self._logits = np.array(logits, dtype=np.float32)
        self._input_name = input_name

    def get_inputs(self):
        return [types.SimpleNamespace(name=self._input_name)]

    def get_outputs(self):
        return [types.SimpleNamespace(shape=[None, self._logits.shape[-1]])]

    def run(self, output_names, feed):
        tensor = feed[self._input_name]
        batch = tensor.shape[0]
        return [np.tile(self._logits, (batch, 1))]


def _imagen_real(path, size=(4, 4)):
    from PIL import Image
    Image.new("RGB", size, color=(10, 20, 30)).save(str(path))


@pytest.fixture(autouse=True)
def _limpiar_cache():
    classifierExe.evict_model_cache()
    yield
    classifierExe.evict_model_cache()


class TestPromedioDeEnsemble:

    def test_promedia_softmax_entre_folds_no_los_logits(self, tmp_path):
        """Dos folds simétricos (logits [2,0] y [0,2]) tienen que promediar a
        exactamente [0.5, 0.5] en probabilidad — si el código promediara los
        logits crudos en vez del softmax, o promediara mal, esto no daría
        medio a medio."""
        model_dir = str(tmp_path / "modelo")
        os.makedirs(model_dir, exist_ok=True)
        img_path = tmp_path / "img1.jpg"
        _imagen_real(img_path)

        meta = {
            "framework": "onnx",
            "image_size": [4, 4],
            "class_names": ["MEDU", "ZOEA"],
            "preprocessing": "identity",
        }

        sesiones = [_FakeSession([[2.0, 0.0]]), _FakeSession([[0.0, 2.0]])]
        cache_key = (classifierExe.model_key(model_dir), False)
        classifierExe._MODEL_CACHE[cache_key] = sesiones

        pred = classifierExe._classify_onnx(model_dir, meta, [str(img_path)], batch_size=16)

        clase, confianza = pred["img1.jpg"]
        assert clase == "MEDU"  # empate 0.5/0.5 -> np.argmax se queda con el primero
        assert confianza == pytest.approx(0.5, abs=1e-5)

    def test_un_solo_fold_da_el_softmax_exacto(self, tmp_path):
        model_dir = str(tmp_path / "modelo")
        os.makedirs(model_dir, exist_ok=True)
        img_path = tmp_path / "img1.jpg"
        _imagen_real(img_path)

        meta = {
            "framework": "onnx",
            "image_size": [4, 4],
            "class_names": ["A", "B", "C"],
            "preprocessing": "identity",
        }

        logits = np.array([1.0, 0.0, 0.0])
        esperado = np.exp(logits) / np.exp(logits).sum()

        cache_key = (classifierExe.model_key(model_dir), False)
        classifierExe._MODEL_CACHE[cache_key] = [_FakeSession([logits.tolist()])]

        pred = classifierExe._classify_onnx(model_dir, meta, [str(img_path)], batch_size=16)

        clase, confianza = pred["img1.jpg"]
        assert clase == "A"
        assert confianza == pytest.approx(esperado[0], abs=1e-6)

    def test_desacuerdo_fuerte_lo_gana_la_mayoria_del_ensemble(self, tmp_path):
        """Tres folds a favor de ZOEA, uno a favor de MEDU: el promedio tiene
        que inclinarse por ZOEA, no por el fold más confiado en soledad."""
        model_dir = str(tmp_path / "modelo")
        os.makedirs(model_dir, exist_ok=True)
        img_path = tmp_path / "img1.jpg"
        _imagen_real(img_path)

        meta = {
            "framework": "onnx",
            "image_size": [4, 4],
            "class_names": ["MEDU", "ZOEA"],
            "preprocessing": "identity",
        }

        sesiones = [
            _FakeSession([[10.0, 0.0]]),  # muy confiado en MEDU, solo
            _FakeSession([[0.0, 1.0]]),
            _FakeSession([[0.0, 1.0]]),
            _FakeSession([[0.0, 1.0]]),
        ]
        cache_key = (classifierExe.model_key(model_dir), False)
        classifierExe._MODEL_CACHE[cache_key] = sesiones

        pred = classifierExe._classify_onnx(model_dir, meta, [str(img_path)], batch_size=16)

        clase, _ = pred["img1.jpg"]
        assert clase == "ZOEA"

    def test_out_dim_distinto_a_class_names_lanza_error_claro(self, tmp_path):
        """meta.json declara 2 clases pero el grafo tiene 3 salidas: las
        etiquetas quedarían corridas sin ningún error visible si no se
        valida."""
        model_dir = str(tmp_path / "modelo")
        os.makedirs(model_dir, exist_ok=True)
        img_path = tmp_path / "img1.jpg"
        _imagen_real(img_path)

        meta = {
            "framework": "onnx",
            "image_size": [4, 4],
            "class_names": ["MEDU", "ZOEA"],  # dice 2
            "preprocessing": "identity",
        }

        cache_key = (classifierExe.model_key(model_dir), False)
        classifierExe._MODEL_CACHE[cache_key] = [_FakeSession([[1.0, 0.0, 0.0]])]  # da 3

        with pytest.raises(ValueError, match="clases"):
            classifierExe._classify_onnx(model_dir, meta, [str(img_path)], batch_size=16)
