"""
Tests de la función "Cancelar" y de la barra de progreso real (en vez de
indeterminada) agregadas a segmentación y clasificación.

Cubre tres capas:
  - classifierExe._classify_onnx: should_stop corta el loop de batches,
    on_progress se llama con (hecho, total) correctos.
  - Segmenter_All_No_View.Segmenter_No_View: mismo contrato, por imagen.
  - Los hilos de la GUI (ClassifierThread, WorkerThread): loguean "Cancelado
    por el usuario" en vez de "❌ Error" cuando se corta, y el botón Cancelar
    llama a requestInterruption() (que es lo que should_stop() consulta).
"""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deepzplankton.core.classifier as classifierExe
from deepzplankton.core.classifier import ProcesoCancelado


# ============================================================
# _classify_onnx: should_stop / on_progress
# ============================================================

class _FakeSession:
    def __init__(self, logits=(1.0, 0.0)):
        self._logits = np.array(logits, dtype=np.float32)
        self.llamadas = 0

    def get_inputs(self):
        return [types.SimpleNamespace(name="input")]

    def get_outputs(self):
        return [types.SimpleNamespace(shape=[None, self._logits.shape[-1]])]

    def run(self, output_names, feed):
        self.llamadas += 1
        tensor = feed["input"]
        return [np.tile(self._logits, (tensor.shape[0], 1))]


def _imagen_real(path, size=(4, 4)):
    from PIL import Image
    Image.new("RGB", size, color=(5, 5, 5)).save(str(path))


@pytest.fixture(autouse=True)
def _limpiar_cache():
    classifierExe.evict_model_cache()
    yield
    classifierExe.evict_model_cache()


def _meta():
    return {
        "framework": "onnx",
        "image_size": [4, 4],
        "class_names": ["MEDU", "ZOEA"],
        "preprocessing": "identity",
    }


class TestClassifyOnnxCancelacion:

    def test_should_stop_true_desde_el_arranque_no_corre_ninguna_inferencia(self, tmp_path):
        model_dir = str(tmp_path / "modelo")
        os.makedirs(model_dir, exist_ok=True)
        img = tmp_path / "a.jpg"
        _imagen_real(img)

        sesion = _FakeSession()
        classifierExe._MODEL_CACHE[(classifierExe.model_key(model_dir), False)] = [sesion]

        with pytest.raises(ProcesoCancelado):
            classifierExe._classify_onnx(
                model_dir, _meta(), [str(img)], batch_size=16, should_stop=lambda: True
            )
        assert sesion.llamadas == 0

    def test_se_corta_entre_batches_y_ya_no_corre_el_siguiente(self, tmp_path):
        model_dir = str(tmp_path / "modelo")
        os.makedirs(model_dir, exist_ok=True)
        imgs = []
        for i in range(3):
            p = tmp_path / f"img{i}.jpg"
            _imagen_real(p)
            imgs.append(str(p))

        sesion = _FakeSession()
        classifierExe._MODEL_CACHE[(classifierExe.model_key(model_dir), False)] = [sesion]

        contador = {"n": 0}
        def should_stop():
            contador["n"] += 1
            return contador["n"] > 1  # deja pasar el primer batch, corta antes del segundo

        with pytest.raises(ProcesoCancelado):
            classifierExe._classify_onnx(
                model_dir, _meta(), imgs, batch_size=1, should_stop=should_stop
            )
        assert sesion.llamadas == 1  # sólo el primer batch llegó a inferir

    def test_on_progress_recibe_hecho_y_total_por_batch(self, tmp_path):
        model_dir = str(tmp_path / "modelo")
        os.makedirs(model_dir, exist_ok=True)
        imgs = []
        for i in range(4):
            p = tmp_path / f"img{i}.jpg"
            _imagen_real(p)
            imgs.append(str(p))

        sesion = _FakeSession()
        classifierExe._MODEL_CACHE[(classifierExe.model_key(model_dir), False)] = [sesion]

        llamadas = []
        classifierExe._classify_onnx(
            model_dir, _meta(), imgs, batch_size=2,
            on_progress=lambda hecho, total: llamadas.append((hecho, total)),
        )
        assert llamadas == [(1, 2), (2, 2)]  # 4 imágenes, batch_size=2 -> 2 batches


# ============================================================
# Segmenter_No_View: should_stop / on_progress
# ============================================================

class TestSegmenterNoViewCancelacion:

    def test_should_stop_true_desde_el_arranque_no_procesa_ninguna_imagen(
        self, tmp_path, monkeypatch
    ):
        import deepzplankton.segmentation.fast as seg
        import deepzplankton.core.paths as app_paths

        monkeypatch.setattr(app_paths, "data_path", lambda *p: str(tmp_path.joinpath(*p)))
        monkeypatch.setattr(seg, "data_path", app_paths.data_path)

        leidas = []
        monkeypatch.setattr(seg.cv2, "imread", lambda p: leidas.append(p))

        with pytest.raises(ProcesoCancelado):
            seg.Segmenter_No_View(["a.jpg", "b.jpg"], should_stop=lambda: True)

        assert leidas == []  # ni siquiera intentó leer la primera

    def test_on_progress_avanza_por_cada_imagen_valida_o_no(self, tmp_path, monkeypatch):
        """Incluye una imagen con extensión inválida: igual cuenta para el
        progreso, o la barra se quedaría trabada antes de llegar al 100%."""
        import deepzplankton.segmentation.fast as seg
        import deepzplankton.core.paths as app_paths

        monkeypatch.setattr(app_paths, "data_path", lambda *p: str(tmp_path.joinpath(*p)))
        monkeypatch.setattr(seg, "data_path", app_paths.data_path)
        monkeypatch.setattr(seg.cv2, "imread", lambda p: None)  # simula lectura fallida

        llamadas = []
        seg.Segmenter_No_View(
            ["a.jpg", "b.txt", "c.jpg"],  # b.txt: extensión no soportada
            on_progress=lambda hecho, total: llamadas.append((hecho, total)),
        )
        assert llamadas == [(1, 3), (2, 3), (3, 3)]


# ============================================================
# GUI: ClassifierThread / WorkerThread ante cancelación
# ============================================================

import deepzplankton.gui.runtime_paths as rt
from deepzplankton.gui.threads.classification_thread import ClassifierThread


class TestClassifierThreadCancelacion:

    def test_cancelado_se_loguea_distinto_de_un_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rt, "ruta_test", str(tmp_path / "Cache"))
        monkeypatch.setattr(rt, "ruta_procesado", str(tmp_path / "Procesado"))
        monkeypatch.setattr(rt, "procesado_excel", str(tmp_path / "Datos" / "Procesado.xlsx"))
        monkeypatch.setattr(rt, "excel_path", str(tmp_path / "Datos" / "Mediciones.xlsx"))
        os.makedirs(str(tmp_path / "Cache"))

        def _fake_classifier(*a, should_stop=None, **k):
            if should_stop and should_stop():
                raise ProcesoCancelado("cancelado")

        monkeypatch.setattr(classifierExe, "classifier", _fake_classifier)

        hilo = ClassifierThread("modelo_x", umbral=0.9)
        # QThread.requestInterruption() sólo prende la bandera con el hilo
        # real corriendo (.start()); acá se corre .run() directo, así que se
        # simula la bandera ya puesta en vez del hilo real de Qt.
        monkeypatch.setattr(hilo, "isInterruptionRequested", lambda: True)

        logs = []
        terminado = []
        hilo.log.connect(logs.append)
        hilo.finished.connect(lambda: terminado.append(True))
        hilo.run()

        assert terminado == [True]
        assert any("Cancelado por el usuario" in l for l in logs)
        assert not any("❌ Error" in l for l in logs)

    def test_progress_se_propaga_desde_el_clasificador(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rt, "ruta_test", str(tmp_path / "Cache"))
        monkeypatch.setattr(rt, "ruta_procesado", str(tmp_path / "Procesado"))
        monkeypatch.setattr(rt, "procesado_excel", str(tmp_path / "Datos" / "Procesado.xlsx"))
        monkeypatch.setattr(rt, "excel_path", str(tmp_path / "Datos" / "Mediciones.xlsx"))
        os.makedirs(str(tmp_path / "Cache"))

        def _fake_classifier(*a, on_progress=None, **k):
            if on_progress:
                on_progress(1, 3)
                on_progress(2, 3)
                on_progress(3, 3)

        monkeypatch.setattr(classifierExe, "classifier", _fake_classifier)

        hilo = ClassifierThread("modelo_x", umbral=0.9)
        avances = []
        hilo.progress.connect(lambda h, t: avances.append((h, t)))
        hilo.run()

        assert avances == [(1, 3), (2, 3), (3, 3)]
