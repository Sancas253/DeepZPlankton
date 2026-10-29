"""
Tests de la lógica de negocio metida en DeepZPlankton_V2.py (la GUI).

DeepZPlankton_V2.py sólo crea un QApplication bajo `if __name__ == "__main__"`,
así que importarlo es seguro. Las clases QThread se pueden instanciar y
correr con `.run()` directo (sin `.start()`, sin thread real, sin event
loop) — es una llamada a método común y corriente; las señales pyqtSignal
emiten igual aunque nadie escuche, y conectarlas a un callback Python
las hace funcionar como un logger sincrónico.

Antes esta lógica (resumen de EnRevision/, y sobre todo ActualizarThread —
el commit final EnRevision/ → Historico/) no tenía ningún test.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deepzplankton.core.filtering as fd_module
import deepzplankton.gui.runtime_paths as rt
from deepzplankton.gui import summary
from deepzplankton.gui.threads.classification_thread import ClassifierThread
from deepzplankton.gui.threads.settle_thread import ActualizarThread


# ============================================================
# _resumen_procesado / _count_procesado
# ============================================================

class TestCountProcesado:

    def test_cuenta_archivos_en_carpetas_de_clase(self, tmp_path, monkeypatch):
        procesado = tmp_path / "Procesado"
        clase_dir = procesado / "MEDU"
        clase_dir.mkdir(parents=True)
        (clase_dir / "a.jpg").write_bytes(b"x")
        (clase_dir / "b.jpg").write_bytes(b"x")
        monkeypatch.setattr(rt, "ruta_procesado", str(procesado))

        assert summary.count_procesado() == 2

    def test_incluye_los_dudosos(self, tmp_path, monkeypatch):
        procesado = tmp_path / "Procesado"
        (procesado / "MEDU").mkdir(parents=True)
        (procesado / "MEDU" / "a.jpg").write_bytes(b"x")
        dud_dir = procesado / "_DUDOSOS" / "ZOEA"
        dud_dir.mkdir(parents=True)
        (dud_dir / "b.jpg").write_bytes(b"x")
        (dud_dir / "c.jpg").write_bytes(b"x")
        monkeypatch.setattr(rt, "ruta_procesado", str(procesado))

        assert summary.count_procesado() == 3

    def test_devuelve_cero_si_no_existe_la_carpeta(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rt, "ruta_procesado", str(tmp_path / "no_existe"))
        assert summary.count_procesado() == 0


class TestResumenProcesado:

    def _excel(self, tmp_path, rows):
        datos = tmp_path / "Datos"
        datos.mkdir()
        path = datos / "Procesado.xlsx"
        pd.DataFrame(rows).to_excel(str(path), index=False)
        return str(path)

    def test_devuelve_vacio_si_no_existe_el_excel(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rt, "procesado_excel", str(tmp_path / "no_existe.xlsx"))
        assert summary.resumen_procesado(0.9) == ""

    def test_devuelve_vacio_si_faltan_columnas(self, tmp_path, monkeypatch):
        path = self._excel(tmp_path, [{"segment": "a.jpg"}])
        monkeypatch.setattr(rt, "procesado_excel", path)
        assert summary.resumen_procesado(0.9) == ""

    def test_cuenta_seguros_y_dudosos_segun_umbral(self, tmp_path, monkeypatch):
        path = self._excel(tmp_path, [
            {"segment": "a.jpg", "class_pred": "MEDU", "confidence": 0.95},
            {"segment": "b.jpg", "class_pred": "MEDU", "confidence": 0.40},
            {"segment": "c.jpg", "class_pred": "ZOEA", "confidence": 0.99},
        ])
        monkeypatch.setattr(rt, "procesado_excel", path)

        resumen = summary.resumen_procesado(0.9)

        assert "Total: 3" in resumen
        assert "Seguros: 2" in resumen
        assert "Dudosos: 1" in resumen
        assert "MEDU: 2" in resumen
        assert "ZOEA: 1" in resumen


# ============================================================
# ActualizarThread: el commit final EnRevision/ -> Historico/
# ============================================================

def _preparar_rutas(tmp_path, monkeypatch):
    ruta_procesado = tmp_path / "Procesado"
    ruta_salida = tmp_path / "Clasificados"
    procesado_excel = tmp_path / "Datos" / "Procesado.xlsx"

    monkeypatch.setattr(rt, "ruta_procesado", str(ruta_procesado))
    monkeypatch.setattr(rt, "ruta_salida", str(ruta_salida))
    monkeypatch.setattr(rt, "procesado_excel", str(procesado_excel))
    # ActualizarThread.run() llama a data_path("Datos", "Historico.xlsx")
    # directo, así que hay que redirigir la función, no una constante.
    monkeypatch.setattr(rt, "data_path", lambda *parts: str(tmp_path.joinpath(*parts)))

    # filtrar_dudosos se prueba aparte (TestFiltrarDudosos en test_bugs.py);
    # acá sólo interesa que ActualizarThread lo llame con el umbral correcto.
    llamadas = []
    monkeypatch.setattr(fd_module, "filtrar_dudosos", lambda umbral: llamadas.append(umbral))

    return ruta_procesado, ruta_salida, procesado_excel, llamadas


def _correr_asentar(umbral):
    hilo = ActualizarThread(umbral)
    logs = []
    hilo.log.connect(logs.append)
    hilo.run()
    return logs


class TestActualizarThread:

    def test_mueve_procesado_a_clasificados_sin_el_prefijo(self, tmp_path, monkeypatch):
        ruta_procesado, ruta_salida, procesado_excel, _ = _preparar_rutas(tmp_path, monkeypatch)

        clase_dir = ruta_procesado / "MEDU"
        clase_dir.mkdir(parents=True)
        (clase_dir / "080_seg1.jpg").write_bytes(b"img")

        procesado_excel.parent.mkdir(exist_ok=True)
        pd.DataFrame({
            "segment": ["seg1.jpg"],
            "class_pred": ["MEDU"],
            "confidence": [0.80],
        }).to_excel(str(procesado_excel), index=False)

        logs = _correr_asentar(0.5)

        destino = ruta_salida / "MEDU" / "seg1.jpg"
        assert destino.exists()
        assert any("asentadas" in l for l in logs)

        clas_excel = pd.read_excel(str(tmp_path / "Datos" / "Historico.xlsx"))
        assert "seg1.jpg" in clas_excel.iloc[:, 0].astype(str).values

    def test_incluye_los_dudosos_reclasificados(self, tmp_path, monkeypatch):
        """Un archivo en _DUDOSOS/<clase>/ se asienta igual que uno normal:
        la clase real es la de la carpeta, no la que diga el Excel viejo."""
        ruta_procesado, ruta_salida, procesado_excel, _ = _preparar_rutas(tmp_path, monkeypatch)

        dud_dir = ruta_procesado / "_DUDOSOS" / "ZOEA"
        dud_dir.mkdir(parents=True)
        (dud_dir / "075_seg2.jpg").write_bytes(b"img")

        procesado_excel.parent.mkdir(exist_ok=True)
        pd.DataFrame({
            "segment": ["seg2.jpg"],
            "class_pred": ["ZOEA"],
            "confidence": [0.75],
        }).to_excel(str(procesado_excel), index=False)

        _correr_asentar(0.9)

        assert (ruta_salida / "ZOEA" / "seg2.jpg").exists()

    def test_elimina_la_version_vieja_en_otra_clase(self, tmp_path, monkeypatch):
        """Si el mismo segmento ya estaba en Clasificados/ bajo otra clase
        (reclasificación), la copia vieja se borra."""
        ruta_procesado, ruta_salida, procesado_excel, _ = _preparar_rutas(tmp_path, monkeypatch)

        vieja_dir = ruta_salida / "OTRA_CLASE"
        vieja_dir.mkdir(parents=True)
        (vieja_dir / "seg1.jpg").write_bytes(b"version vieja")

        clase_dir = ruta_procesado / "MEDU"
        clase_dir.mkdir(parents=True)
        (clase_dir / "090_seg1.jpg").write_bytes(b"version nueva")

        procesado_excel.parent.mkdir(exist_ok=True)
        pd.DataFrame({
            "segment": ["seg1.jpg"],
            "class_pred": ["MEDU"],
            "confidence": [0.90],
        }).to_excel(str(procesado_excel), index=False)

        _correr_asentar(0.5)

        assert not (vieja_dir / "seg1.jpg").exists()
        assert (ruta_salida / "MEDU" / "seg1.jpg").exists()

    def test_reclasificar_a_mano_asienta_confidence_1(self, tmp_path, monkeypatch):
        """Si el usuario movió la imagen a otra carpeta de clase, al asentar
        la fila queda con la clase de la carpeta Y confidence = 1.0: es una
        decisión humana, no la vuelve a agarrar filtrar_dudosos()."""
        ruta_procesado, ruta_salida, procesado_excel, _ = _preparar_rutas(tmp_path, monkeypatch)

        # EnRevision.xlsx dice ZOEA con confianza baja; el usuario la movió a MEDU/
        clase_dir = ruta_procesado / "MEDU"
        clase_dir.mkdir(parents=True)
        (clase_dir / "040_seg1.jpg").write_bytes(b"img")

        procesado_excel.parent.mkdir(exist_ok=True)
        pd.DataFrame({
            "segment": ["seg1.jpg"],
            "class_pred": ["ZOEA"],
            "confidence": [0.40],
        }).to_excel(str(procesado_excel), index=False)

        _correr_asentar(0.9)

        clas_excel = pd.read_excel(str(tmp_path / "Datos" / "Historico.xlsx"))
        fila = clas_excel[clas_excel.iloc[:, 0].astype(str) == "seg1.jpg"].iloc[0]
        assert fila["class_pred"] == "MEDU"
        assert fila["confidence"] == 1.0

    def test_sin_reclasificar_no_toca_confidence(self, tmp_path, monkeypatch):
        """Si la imagen sigue en la carpeta de su clase original, asentar deja
        la confianza del modelo intacta (no la fuerza a 1.0)."""
        ruta_procesado, ruta_salida, procesado_excel, _ = _preparar_rutas(tmp_path, monkeypatch)

        clase_dir = ruta_procesado / "MEDU"
        clase_dir.mkdir(parents=True)
        (clase_dir / "072_seg1.jpg").write_bytes(b"img")

        procesado_excel.parent.mkdir(exist_ok=True)
        pd.DataFrame({
            "segment": ["seg1.jpg"],
            "class_pred": ["MEDU"],
            "confidence": [0.72],
        }).to_excel(str(procesado_excel), index=False)

        _correr_asentar(0.5)

        clas_excel = pd.read_excel(str(tmp_path / "Datos" / "Historico.xlsx"))
        fila = clas_excel[clas_excel.iloc[:, 0].astype(str) == "seg1.jpg"].iloc[0]
        assert fila["class_pred"] == "MEDU"
        assert fila["confidence"] == 0.72

    def test_vacia_procesado_y_llama_a_filtrar_dudosos_con_el_umbral(self, tmp_path, monkeypatch):
        ruta_procesado, ruta_salida, procesado_excel, llamadas = _preparar_rutas(tmp_path, monkeypatch)

        clase_dir = ruta_procesado / "MEDU"
        clase_dir.mkdir(parents=True)
        (clase_dir / "080_seg1.jpg").write_bytes(b"img")

        procesado_excel.parent.mkdir(exist_ok=True)
        pd.DataFrame({
            "segment": ["seg1.jpg"], "class_pred": ["MEDU"], "confidence": [0.80],
        }).to_excel(str(procesado_excel), index=False)

        _correr_asentar(0.63)

        assert llamadas == [0.63]
        assert ruta_procesado.exists()
        assert list(ruta_procesado.iterdir()) == []  # recreada vacía
        assert not procesado_excel.exists()

    def test_enrevision_vacio_salta_asentar_pero_igual_reconcilia(self, tmp_path, monkeypatch):
        """Sin nada en EnRevision/ el botón no asienta (no toca filtrar_dudosos)
        pero SÍ corre la reconciliación de Historico/ — es el caso "solo
        actualizar" del botón unificado."""
        _, _, _, llamadas = _preparar_rutas(tmp_path, monkeypatch)
        # Ni EnRevision/ ni EnRevision.xlsx existen.

        logs = _correr_asentar(0.9)

        assert any("vac" in l.lower() for l in logs)   # avisa que EnRevision está vacío
        assert any("reconciliando Historico" in l for l in logs)
        assert llamadas == []  # no llega a filtrar_dudosos (nada para asentar)

    def test_reconcilia_historico_aunque_no_haya_tanda(self, tmp_path, monkeypatch):
        """EnRevision/ vacío, pero el usuario movió a mano un archivo en
        Historico/ a otra clase: al apretar "Asentar" la parte de actualizar
        retagea el Excel y le pone confidence = 1.0."""
        _, ruta_salida, _, _ = _preparar_rutas(tmp_path, monkeypatch)

        # Historico/ con seg_x físicamente en CHAE, pero los Excel dicen CLAD
        (ruta_salida / "CHAE").mkdir(parents=True)
        (ruta_salida / "CHAE" / "seg_x.jpg").write_bytes(b"img")

        datos = tmp_path / "Datos"
        datos.mkdir(exist_ok=True)
        for nombre in ("Historico.xlsx", "Seguros.xlsx"):
            pd.DataFrame({
                "segment": ["seg_x.jpg"], "class_pred": ["CLAD"], "confidence": [0.99],
            }).to_excel(str(datos / nombre), index=False)
        pd.DataFrame(columns=["segment", "class_pred", "confidence"]).to_excel(
            str(datos / "Dudosos.xlsx"), index=False
        )

        _correr_asentar(0.9)

        hist = pd.read_excel(str(datos / "Historico.xlsx"))
        fila = hist[hist.iloc[:, 0].astype(str) == "seg_x.jpg"].iloc[0]
        assert fila["class_pred"] == "CHAE"
        assert fila["confidence"] == 1.0

    def test_un_error_no_deja_de_emitir_finished(self, tmp_path, monkeypatch):
        """El try/except de run() tiene que emitir finished pase lo que
        pase, o la barra de progreso de la GUI se queda tildada."""
        ruta_procesado, ruta_salida, procesado_excel, _ = _preparar_rutas(tmp_path, monkeypatch)

        clase_dir = ruta_procesado / "MEDU"
        clase_dir.mkdir(parents=True)
        (clase_dir / "080_seg1.jpg").write_bytes(b"img")
        procesado_excel.parent.mkdir(exist_ok=True)
        pd.DataFrame({
            "segment": ["seg1.jpg"], "class_pred": ["MEDU"], "confidence": [0.80],
        }).to_excel(str(procesado_excel), index=False)

        def _explota(umbral):
            raise RuntimeError("boom")
        monkeypatch.setattr(fd_module, "filtrar_dudosos", _explota)

        hilo = ActualizarThread(0.9)
        logs = []
        terminado = []
        hilo.log.connect(logs.append)
        hilo.finished.connect(lambda: terminado.append(True))

        hilo.run()

        assert terminado == [True]
        assert any("Error al asentar" in l for l in logs)


# ============================================================
# ClassifierThread
# ============================================================

class TestClassifierThread:

    def _preparar(self, tmp_path, monkeypatch):
        ruta_test = tmp_path / "Cache"
        ruta_procesado = tmp_path / "Procesado"
        procesado_excel = tmp_path / "Datos" / "Procesado.xlsx"
        excel_path = tmp_path / "Datos" / "Mediciones.xlsx"
        monkeypatch.setattr(rt, "ruta_test", str(ruta_test))
        monkeypatch.setattr(rt, "ruta_procesado", str(ruta_procesado))
        monkeypatch.setattr(rt, "procesado_excel", str(procesado_excel))
        monkeypatch.setattr(rt, "excel_path", str(excel_path))
        return ruta_test, ruta_procesado, procesado_excel, excel_path

    def test_usa_las_imagenes_ya_presentes_en_el_cache_si_no_arrastran_nada(
        self, tmp_path, monkeypatch
    ):
        ruta_test, ruta_procesado, procesado_excel, excel_path = self._preparar(tmp_path, monkeypatch)
        ruta_test.mkdir(parents=True)

        llamadas = []

        def _fake_classifier(modelo, ruta_test_arg, ruta_salida_arg, excel_arg,
                              best_fold_only=False, output_excel_path=None,
                              on_progress=None, should_stop=None, umbral=None):
            llamadas.append((modelo, ruta_test_arg, best_fold_only))

        import deepzplankton.core.classifier as classifierExe
        monkeypatch.setattr(classifierExe, "classifier", _fake_classifier)

        hilo = ClassifierThread("modelo_x", umbral=0.9, imagenes=None, best_fold_only=True)
        logs = []
        hilo.log.connect(logs.append)
        hilo.run()

        assert llamadas == [("modelo_x", str(ruta_test), True)]
        assert any("finalizada" in l for l in logs)

    def test_copia_las_imagenes_arrastradas_al_cache_sin_enlazar(self, tmp_path, monkeypatch):
        """Las imágenes que arrastra el usuario son suyas: se copian de
        verdad, no se enlazan (ver comentario en ClassifierThread.run())."""
        ruta_test, ruta_procesado, procesado_excel, excel_path = self._preparar(tmp_path, monkeypatch)

        origen = tmp_path / "afuera"
        origen.mkdir()
        img = origen / "mia.jpg"
        img.write_bytes(b"original")

        import deepzplankton.core.classifier as classifierExe
        monkeypatch.setattr(classifierExe, "classifier", lambda *a, **k: None)

        hilo = ClassifierThread("modelo_x", umbral=0.9, imagenes=[str(img)])
        hilo.run()

        copia = ruta_test / "mia.jpg"
        assert copia.exists()
        # No es el mismo inode/enlace: modificar la copia no debe tocar el original.
        copia.write_bytes(b"modificada")
        assert img.read_bytes() == b"original"

    def test_un_error_del_clasificador_se_loguea_y_no_crashea(self, tmp_path, monkeypatch):
        ruta_test, ruta_procesado, procesado_excel, excel_path = self._preparar(tmp_path, monkeypatch)
        ruta_test.mkdir(parents=True)

        import deepzplankton.core.classifier as classifierExe

        def _explota(*a, **k):
            raise RuntimeError("modelo corrupto")
        monkeypatch.setattr(classifierExe, "classifier", _explota)

        hilo = ClassifierThread("modelo_x", umbral=0.9)
        logs = []
        terminado = []
        hilo.log.connect(logs.append)
        hilo.finished.connect(lambda: terminado.append(True))

        hilo.run()

        assert terminado == [True]
        assert any("Error" in l and "modelo corrupto" in l for l in logs)
