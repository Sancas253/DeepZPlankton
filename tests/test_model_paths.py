"""
Una sola resolución de rutas de modelo.

Antes había dos: app_paths.resource_path() (relativa al bundle, con la carpeta
APP_NAME adentro) y una segunda resource_path() en classifierExe.py que fuera
del bundle devolvía la ruta tal cual, o sea relativa al directorio de trabajo.
No eran intercambiables y se usaban para partes distintas del MISMO modelo:
load_meta() abría model_dir/meta.json directo, mientras _load_onnx_sessions()
pasaba model_dir por resource_path(). Con una ruta relativa en el .exe eso
significa leer el manifest de un lado y los pesos del otro.
"""
import json
import os

import pytest

import deepzplankton.core.paths as app_paths
import deepzplankton.core.classifier as classifierExe
from deepzplankton.core.paths import model_key, model_path


def test_classifier_no_define_su_propia_resolucion():
    assert not hasattr(classifierExe, "resource_path")


def test_model_path_devuelve_absoluta(tmp_path, monkeypatch):
    monkeypatch.setattr(app_paths, "app_dir", lambda: str(tmp_path))
    resuelta = model_path("mimodelo")
    assert os.path.isabs(resuelta)
    assert resuelta == os.path.join(str(tmp_path), "mimodelo")


def test_model_path_es_idempotente(tmp_path, monkeypatch):
    monkeypatch.setattr(app_paths, "app_dir", lambda: str(tmp_path))
    una = model_path("mimodelo")
    assert model_path(una) == una


def test_model_path_respeta_las_absolutas(tmp_path):
    """El combo pasa rutas absolutas: no deben reubicarse dentro del bundle."""
    afuera = str(tmp_path / "en_otro_disco")
    assert model_path(afuera) == os.path.normpath(afuera)


def test_meta_y_pesos_se_resuelven_igual(tmp_path, monkeypatch):
    """La regresión: con model_dir relativo, el manifest tiene que salir del
    mismo directorio del que saldrían los pesos, no del cwd."""
    bundle = tmp_path / "bundle"
    (bundle / "mimodelo").mkdir(parents=True)
    (bundle / "mimodelo" / "meta.json").write_text(
        json.dumps({
            "framework": "onnx",
            "image_size": [224, 224],
            "class_names": ["A", "B"],
            "preprocessing": "imagenet_meanstd",
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_paths, "app_dir", lambda: str(bundle))

    cwd = tmp_path / "otro_lado"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    meta = classifierExe.load_meta("mimodelo")
    assert meta["class_names"] == ["A", "B"]


def test_load_meta_falla_si_el_modelo_no_existe(tmp_path, monkeypatch):
    monkeypatch.setattr(app_paths, "app_dir", lambda: str(tmp_path))
    with pytest.raises(FileNotFoundError):
        classifierExe.load_meta("no_existe")


def test_model_key_ignora_mayusculas_en_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(app_paths, "app_dir", lambda: str(tmp_path))
    if os.path.normcase("A") == os.path.normcase("a"):
        assert model_key("MiModelo") == model_key("mimodelo")
    else:
        pytest.skip("sistema de archivos sensible a mayúsculas")


def test_evict_acepta_la_ruta_escrita_de_otra_forma(tmp_path, monkeypatch):
    """Reemplazar un modelo tiene que tirar el caché aunque la ruta llegue
    normalizada distinto de como se guardó."""
    monkeypatch.setattr(app_paths, "app_dir", lambda: str(tmp_path))
    directo = str(tmp_path / "mimodelo")
    rodeo = str(tmp_path / "otro" / ".." / "mimodelo")

    classifierExe._MODEL_CACHE.clear()
    classifierExe._MODEL_CACHE[(model_key(directo), False)] = ["sesion"]
    classifierExe.evict_model_cache(rodeo)
    assert classifierExe._MODEL_CACHE == {}
