"""
Tests de la carga de sesiones ONNX.

La regresión que motivó el archivo: _load_onnx_sessions() asignaba a una local
llamada `model_path`, que es también la función importada de app_paths que la
propia función usa en su primera línea para resolver el directorio. Python trata
el nombre como local en todo el cuerpo, así que esa primera llamada moría con
UnboundLocalError y NINGÚN modelo cargaba — ni ensemble ni model.onnx único.

No lo cazaba nadie porque el resto de la suite prueba manifest, rutas y caché,
que se resuelven antes de tocar onnxruntime. Acá se stubea onnxruntime para
poder ejecutar la función de punta a punta sin pesos reales.
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deepzplankton.core.classifier as classifierExe


class _FakeSession:
    def __init__(self, path, providers=None):
        self.path = path
        self.providers = providers


@pytest.fixture
def onnxruntime_stub(monkeypatch):
    """onnxruntime falso: registra qué archivos se abrieron, sin cargar pesos."""
    stub = types.ModuleType("onnxruntime")
    stub.get_available_providers = lambda: ["CPUExecutionProvider"]
    stub.InferenceSession = _FakeSession
    monkeypatch.setitem(sys.modules, "onnxruntime", stub)
    return stub


def _ensemble(tmp_path, n=5):
    d = tmp_path / "modelo_ensemble"
    d.mkdir()
    for i in range(n):
        (d / f"fold_{i}.onnx").write_bytes(b"onnx")
    return str(d)


def _unico(tmp_path):
    d = tmp_path / "modelo_unico"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"onnx")
    return str(d)


def test_carga_el_ensemble_completo(tmp_path, onnxruntime_stub):
    sesiones = classifierExe._load_onnx_sessions(_ensemble(tmp_path))
    assert len(sesiones) == 5
    assert [os.path.basename(s.path) for s in sesiones] == [
        f"fold_{i}.onnx" for i in range(5)
    ]


def test_carga_el_modelo_unico(tmp_path, onnxruntime_stub):
    """El camino de model.onnx es justo el que tenía la asignación culpable."""
    sesiones = classifierExe._load_onnx_sessions(_unico(tmp_path))
    assert len(sesiones) == 1
    assert os.path.basename(sesiones[0].path) == "model.onnx"


def test_best_fold_only_carga_un_solo_fold(tmp_path, onnxruntime_stub):
    sesiones = classifierExe._load_onnx_sessions(
        _ensemble(tmp_path), best_fold_only=True, best_fold=3
    )
    assert len(sesiones) == 1
    assert os.path.basename(sesiones[0].path) == "fold_3.onnx"


def test_sin_best_fold_en_el_meta_cae_al_ensemble(tmp_path, onnxruntime_stub):
    """El checkbox puede estar tildado con un meta.json que no declara best_fold."""
    sesiones = classifierExe._load_onnx_sessions(
        _ensemble(tmp_path), best_fold_only=True, best_fold=None
    )
    assert len(sesiones) == 5


def test_best_fold_inexistente_cae_al_ensemble(tmp_path, onnxruntime_stub):
    sesiones = classifierExe._load_onnx_sessions(
        _ensemble(tmp_path, n=3), best_fold_only=True, best_fold=9
    )
    assert len(sesiones) == 3


def test_los_folds_ganan_sobre_model_onnx(tmp_path, onnxruntime_stub):
    """Un directorio con las dos cosas es un ensemble, no un modelo único."""
    d = tmp_path / "mixto"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"onnx")
    (d / "fold_0.onnx").write_bytes(b"onnx")
    (d / "fold_1.onnx").write_bytes(b"onnx")
    sesiones = classifierExe._load_onnx_sessions(str(d))
    assert [os.path.basename(s.path) for s in sesiones] == ["fold_0.onnx", "fold_1.onnx"]


def test_no_deja_de_resolver_la_ruta_del_modelo(tmp_path, onnxruntime_stub, monkeypatch):
    """La ruta relativa se resuelve con app_paths.model_path, no contra el cwd."""
    import deepzplankton.core.paths as app_paths

    bundle = tmp_path / "bundle"
    (bundle / "mimodelo").mkdir(parents=True)
    (bundle / "mimodelo" / "model.onnx").write_bytes(b"onnx")
    monkeypatch.setattr(app_paths, "app_dir", lambda: str(bundle))

    otro_lado = tmp_path / "otro_lado"
    otro_lado.mkdir()
    monkeypatch.chdir(otro_lado)

    sesiones = classifierExe._load_onnx_sessions("mimodelo")
    assert sesiones[0].path == os.path.join(str(bundle), "mimodelo", "model.onnx")
