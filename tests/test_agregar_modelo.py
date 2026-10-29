"""
Tests de la instalación de un modelo (botón "Agregar modelo").

Dos regresiones:

  - El destino se borraba antes de copiar. Un modelo son cientos de MB: si la
    copia se cortaba a mitad, el que ya estaba instalado quedaba destruido y el
    nuevo incompleto.
  - El nombre salía siempre de la carpeta origen. Para un ZIP cuyos archivos
    están sueltos en la raíz, esa carpeta es el temporal, así que el modelo
    quedaba instalado y listado como "deepz_model_ab12cd".
"""
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepzplankton.gui.main_window import MainWindow
from deepzplankton.gui.threads.add_model_thread import AgregarModeloThread


def _modelo(base, nombre, contenido=b"onnx"):
    d = base / nombre
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(contenido)
    (d / "meta.json").write_text("{}", encoding="utf-8")
    return d


class TestNombreDeInstalacion:

    def test_por_defecto_usa_el_de_la_carpeta_origen(self, tmp_path):
        src = _modelo(tmp_path / "origen", "swin_ensemble")
        hilo = AgregarModeloThread(str(src), str(tmp_path / "model"))
        assert hilo.nombre == "swin_ensemble"

    def test_ignora_la_barra_final(self, tmp_path):
        src = _modelo(tmp_path / "origen", "swin_ensemble")
        hilo = AgregarModeloThread(str(src) + os.sep, str(tmp_path / "model"))
        assert hilo.nombre == "swin_ensemble"

    def test_el_nombre_explicito_gana(self, tmp_path):
        """Es el caso del ZIP sin carpeta raíz: el origen es un tempdir."""
        src = _modelo(tmp_path, "deepz_model_ab12cd")
        hilo = AgregarModeloThread(str(src), str(tmp_path / "model"),
                                       nombre="efficientnetb3_ensemble")
        assert hilo.nombre == "efficientnetb3_ensemble"


class TestCopia:

    def test_instala_el_modelo(self, tmp_path):
        src = _modelo(tmp_path / "origen", "modelo")
        dest_base = tmp_path / "model"
        dest_base.mkdir()

        AgregarModeloThread(str(src), str(dest_base)).run()

        assert (dest_base / "modelo" / "model.onnx").read_bytes() == b"onnx"

    def test_reemplaza_al_anterior(self, tmp_path):
        _modelo(tmp_path / "model", "modelo", contenido=b"viejo")
        src = _modelo(tmp_path / "origen", "modelo", contenido=b"nuevo")

        AgregarModeloThread(str(src), str(tmp_path / "model")).run()

        assert (tmp_path / "model" / "modelo" / "model.onnx").read_bytes() == b"nuevo"

    def test_una_copia_fallida_no_destruye_el_modelo_anterior(self, tmp_path, monkeypatch):
        _modelo(tmp_path / "model", "modelo", contenido=b"viejo")
        src = _modelo(tmp_path / "origen", "modelo", contenido=b"nuevo")

        def _explota(*a, **k):
            raise OSError("disco lleno")
        monkeypatch.setattr(shutil, "copytree", _explota)

        AgregarModeloThread(str(src), str(tmp_path / "model")).run()

        assert (tmp_path / "model" / "modelo" / "model.onnx").read_bytes() == b"viejo"

    def test_no_deja_restos_de_la_copia_fallida(self, tmp_path, monkeypatch):
        src = _modelo(tmp_path / "origen", "modelo")
        dest_base = tmp_path / "model"
        dest_base.mkdir()

        real_copytree = shutil.copytree

        def _copia_y_explota(origen, destino, *a, **k):
            real_copytree(origen, destino, *a, **k)
            raise OSError("se cortó después de copiar")
        monkeypatch.setattr(shutil, "copytree", _copia_y_explota)

        AgregarModeloThread(str(src), str(dest_base)).run()

        assert os.listdir(dest_base) == []

    def test_limpia_el_temporal_del_zip(self, tmp_path):
        tmpdir = tmp_path / "deepz_model_ab12cd"
        src = _modelo(tmpdir, "modelo")
        dest_base = tmp_path / "model"
        dest_base.mkdir()

        AgregarModeloThread(str(src), str(dest_base), tmpdir=str(tmpdir)).run()

        assert not tmpdir.exists()


class TestModelosInstalados:
    """A qué carpetas hay que pedirles meta.json después de copiar."""

    def test_una_carpeta_de_modelo_es_ella_misma(self, tmp_path):
        src = _modelo(tmp_path, "modelo")
        assert MainWindow._modelos_instalados(str(src)) == [str(src)]

    def test_una_carpeta_padre_devuelve_los_hijos(self, tmp_path):
        padre = tmp_path / "varios"
        _modelo(padre, "swin")
        _modelo(padre, "convnext")

        encontrados = MainWindow._modelos_instalados(str(padre))

        assert sorted(os.path.basename(p) for p in encontrados) == ["convnext", "swin"]

    def test_una_carpeta_sin_modelos_no_devuelve_nada(self, tmp_path):
        vacia = tmp_path / "vacia"
        vacia.mkdir()
        assert MainWindow._modelos_instalados(str(vacia)) == []
