"""
Tests del prefijo de confianza en EnRevision/ (antes "Procesado/").

Revisar es mirar primero lo que el modelo menos tuvo claro, y eso se hace en el
explorador de Windows. Con la confianza adelante del nombre ("048_X.jpg"),
ordenar por nombre es ordenar por duda.

La regla que sostiene todo: el prefijo existe SÓLO en EnRevision/. El nombre real
es la clave con la que se cruzan Mediciones.xlsx, EnRevision.xlsx,
Historico.xlsx y Historico/; si el prefijo se filtra a cualquiera de esos
lados se rompen el asentado, filtrar_dudosos y actualizar_clasificados.
"""
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deepzplankton.core.classifier as classifierExe
from deepzplankton.core.classifier import con_confianza, sin_confianza


# ============================================================
# El formato del prefijo
# ============================================================

class TestFormato:

    def test_usa_tres_digitos_con_ceros_adelante(self):
        assert con_confianza("X.jpg", 0.4812) == "048_X.jpg"
        assert con_confianza("X.jpg", 0.07) == "007_X.jpg"

    def test_la_confianza_maxima_entra_en_tres_digitos(self):
        assert con_confianza("X.jpg", 1.0) == "100_X.jpg"

    def test_solo_el_100_real_se_muestra_como_100(self):
        """"Reviso todo lo que no diga 100" no puede saltearse un 0.9958."""
        assert con_confianza("X.jpg", 0.9958) == "099_X.jpg"
        assert con_confianza("X.jpg", 0.999999) == "099_X.jpg"

    def test_el_redondeo_no_sufre_el_binario(self):
        """0.61*100 da 60.999… : truncar daría 060."""
        assert con_confianza("X.jpg", 0.61) == "061_X.jpg"
        assert con_confianza("X.jpg", 0.29) == "029_X.jpg"

    def test_el_orden_alfabetico_es_el_orden_por_confianza(self):
        """Es todo el punto: el explorador ordena por nombre, nada más."""
        nombres = [con_confianza(f"seg_{i}.jpg", c)
                   for i, c in enumerate([0.99, 0.07, 1.0, 0.5, 0.093])]
        ordenados = sorted(nombres)
        confianzas = [int(n[:3]) for n in ordenados]
        assert confianzas == sorted(confianzas)
        assert ordenados[0].startswith("007_")   # lo más dudoso primero
        assert ordenados[-1].startswith("100_")

    def test_es_idempotente(self):
        """Reclasificar no puede apilar prefijos: 048_061_X.jpg."""
        una = con_confianza("X.jpg", 0.61)
        assert con_confianza(una, 0.48) == "048_X.jpg"

    def test_no_se_come_los_digitos_de_un_nombre_real(self):
        """Con el conjunto de nombres verdaderos delante, no hay ambigüedad."""
        reales = {"001_muestra_3.jpg"}
        assert con_confianza("001_muestra_3.jpg", 0.42, reales) == "042_001_muestra_3.jpg"


class TestSinConfianza:

    def test_saca_el_prefijo(self):
        assert sin_confianza("048_EH0222_12.jpg") == "EH0222_12.jpg"

    def test_deja_intacto_un_nombre_sin_prefijo(self):
        assert sin_confianza("EH0222_12.jpg") == "EH0222_12.jpg"

    def test_saca_un_solo_prefijo(self):
        assert sin_confianza("048_061_X.jpg") == "061_X.jpg"

    def test_el_conjunto_de_nombres_reales_manda(self):
        """Si la imagen de campaña se llamaba 001_muestra.tif, el segmento se
        llama 001_muestra_3.jpg y esos dígitos NO son un prefijo."""
        reales = {"001_muestra_3.jpg"}
        assert sin_confianza("001_muestra_3.jpg", reales) == "001_muestra_3.jpg"

    def test_igual_saca_el_prefijo_de_los_demas(self, ):
        reales = {"001_muestra_3.jpg", "otro_1.jpg"}
        assert sin_confianza("048_otro_1.jpg", reales) == "otro_1.jpg"

    def test_no_toca_nombres_de_otra_forma(self):
        assert sin_confianza("12_X.jpg") == "12_X.jpg"      # dos dígitos
        assert sin_confianza("0481_X.jpg") == "0481_X.jpg"  # cuatro


# ============================================================
# classifier(): escribe con prefijo, informa sin prefijo
# ============================================================

def _modelo(tmp_path):
    d = tmp_path / "modelo"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({
        "framework": "onnx",
        "image_size": [224, 224],
        "class_names": ["MEDU", "ZOEA"],
        "preprocessing": "imagenet_meanstd",
    }), encoding="utf-8")
    (d / "model.onnx").write_bytes(b"onnx")
    return str(d)


def _cache(tmp_path, nombres):
    d = tmp_path / "cache"
    d.mkdir()
    for n in nombres:
        (d / n).write_bytes(b"jpg")
    return str(d)


@pytest.fixture
def predicciones(monkeypatch):
    """Reemplaza la inferencia: acá se prueba el nombrado, no el modelo."""
    def _set(pred):
        monkeypatch.setattr(classifierExe, "_classify_onnx",
                            lambda *a, **k: dict(pred))
    return _set


def _correr(tmp_path, model_dir, cache, procesado, pred_por_defecto=None):
    return classifierExe.classifier(
        model_dir, cache, str(procesado),
        str(tmp_path / "Datos" / "Mediciones.xlsx"),
        output_excel_path=str(tmp_path / "Datos" / "Procesado.xlsx"),
    )


class TestClassifierEscribeElPrefijo:

    def test_los_archivos_de_procesado_llevan_la_confianza(self, tmp_path, predicciones):
        predicciones({"a.jpg": ("MEDU", 0.4812), "b.jpg": ("ZOEA", 0.9987)})
        procesado = tmp_path / "Procesado"

        _correr(tmp_path, _modelo(tmp_path), _cache(tmp_path, ["a.jpg", "b.jpg"]), procesado)

        assert os.listdir(procesado / "MEDU") == ["048_a.jpg"]
        assert os.listdir(procesado / "ZOEA") == ["099_b.jpg"]   # 0.9987, no 1.0

    def test_el_excel_guarda_el_nombre_real(self, tmp_path, predicciones):
        """Si el prefijo entra al Excel, se rompe el cruce con Clasificados."""
        predicciones({"a.jpg": ("MEDU", 0.4812)})

        _correr(tmp_path, _modelo(tmp_path), _cache(tmp_path, ["a.jpg"]),
                tmp_path / "Procesado")

        df = pd.read_excel(tmp_path / "Datos" / "Procesado.xlsx")
        assert list(df["segment"]) == ["a.jpg"]

    def test_reclasificar_con_otro_modelo_no_deja_duplicados(self, tmp_path, predicciones):
        """La copia vieja tiene otro prefijo: hay que borrarla igual."""
        modelo = _modelo(tmp_path)
        cache = _cache(tmp_path, ["a.jpg"])
        procesado = tmp_path / "Procesado"

        predicciones({"a.jpg": ("MEDU", 0.61)})
        _correr(tmp_path, modelo, cache, procesado)
        predicciones({"a.jpg": ("MEDU", 0.48)})
        _correr(tmp_path, modelo, cache, procesado)

        assert os.listdir(procesado / "MEDU") == ["048_a.jpg"]

    def test_reclasificar_a_otra_clase_no_deja_la_copia_vieja(self, tmp_path, predicciones):
        modelo = _modelo(tmp_path)
        cache = _cache(tmp_path, ["a.jpg"])
        procesado = tmp_path / "Procesado"

        predicciones({"a.jpg": ("MEDU", 0.61)})
        _correr(tmp_path, modelo, cache, procesado)
        predicciones({"a.jpg": ("ZOEA", 0.77)})
        _correr(tmp_path, modelo, cache, procesado)

        assert os.listdir(procesado / "MEDU") == []
        assert os.listdir(procesado / "ZOEA") == ["077_a.jpg"]

    def test_con_umbral_los_dudosos_van_a_su_propia_carpeta(self, tmp_path, predicciones):
        """La separación física en Procesado/_DUDOSOS/CLASE, además del
        prefijo, para que se note a simple vista en el explorador."""
        predicciones({"a.jpg": ("MEDU", 0.48), "b.jpg": ("ZOEA", 0.99)})
        procesado = tmp_path / "Procesado"

        classifierExe.classifier(
            _modelo(tmp_path), _cache(tmp_path, ["a.jpg", "b.jpg"]),
            str(procesado), str(tmp_path / "Datos" / "Mediciones.xlsx"),
            output_excel_path=str(tmp_path / "Datos" / "Procesado.xlsx"),
            umbral=0.95,
        )

        assert os.listdir(procesado / "_DUDOSOS" / "MEDU") == ["048_a.jpg"]
        assert os.listdir(procesado / "ZOEA") == ["099_b.jpg"]
        assert not (procesado / "MEDU").exists()

    def test_reclasificar_saca_de_dudosos_si_ya_no_lo_es(self, tmp_path, predicciones):
        """Al reclasificar, la copia vieja en _DUDOSOS se borra si el nuevo
        resultado ya no es dudoso (y viceversa)."""
        modelo = _modelo(tmp_path)
        cache = _cache(tmp_path, ["a.jpg"])
        procesado = tmp_path / "Procesado"

        predicciones({"a.jpg": ("MEDU", 0.48)})
        classifierExe.classifier(
            modelo, cache, str(procesado), str(tmp_path / "Datos" / "Mediciones.xlsx"),
            output_excel_path=str(tmp_path / "Datos" / "Procesado.xlsx"), umbral=0.95,
        )
        predicciones({"a.jpg": ("MEDU", 0.97)})
        classifierExe.classifier(
            modelo, cache, str(procesado), str(tmp_path / "Datos" / "Mediciones.xlsx"),
            output_excel_path=str(tmp_path / "Datos" / "Procesado.xlsx"), umbral=0.95,
        )

        assert os.listdir(procesado / "MEDU") == ["097_a.jpg"]
        assert not (procesado / "_DUDOSOS" / "MEDU" / "048_a.jpg").exists()

    def test_un_segmento_con_digitos_propios_sobrevive(self, tmp_path, predicciones):
        """Imagen de campaña "001_muestra.tif" → segmento "001_muestra_3.jpg"."""
        predicciones({"001_muestra_3.jpg": ("MEDU", 0.42)})
        procesado = tmp_path / "Procesado"

        _correr(tmp_path, _modelo(tmp_path),
                _cache(tmp_path, ["001_muestra_3.jpg"]), procesado)

        assert os.listdir(procesado / "MEDU") == ["042_001_muestra_3.jpg"]
        df = pd.read_excel(tmp_path / "Datos" / "Procesado.xlsx")
        assert list(df["segment"]) == ["001_muestra_3.jpg"]


# ============================================================
# Asentar: el prefijo no puede llegar a Clasificados/
# ============================================================

class TestAsentarSacaElPrefijo:

    @pytest.fixture
    def entorno(self, tmp_path, monkeypatch):
        """Monta Datos/, EnRevision/ y Historico/ dentro de tmp_path."""
        import deepzplankton.core.filtering as fd
        import deepzplankton.gui.runtime_paths as rt
        from deepzplankton.gui.threads.settle_thread import ActualizarThread

        datos = tmp_path / "Datos"
        datos.mkdir()
        procesado = tmp_path / "EnRevision"
        clasificados = tmp_path / "Historico"

        monkeypatch.setattr(rt, "ruta_procesado", str(procesado))
        monkeypatch.setattr(rt, "ruta_salida", str(clasificados))
        monkeypatch.setattr(rt, "procesado_excel", str(datos / "EnRevision.xlsx"))
        monkeypatch.setattr(rt, "data_path",
                            lambda *p: os.path.join(str(tmp_path), *p))
        monkeypatch.setattr(fd, "CLASIFICADOS_EXCEL", str(datos / "Historico.xlsx"))
        monkeypatch.setattr(fd, "SEGUROS_EXCEL", str(datos / "Seguros.xlsx"))
        monkeypatch.setattr(fd, "DUDOSOS_EXCEL", str(datos / "Dudosos.xlsx"))
        monkeypatch.setattr(fd, "CLASIFICADOS_DIR", str(clasificados))
        return ActualizarThread, tmp_path, datos, procesado, clasificados

    def _procesado_con(self, procesado, datos, archivos, filas):
        for clase, nombre in archivos:
            d = procesado / clase
            d.mkdir(parents=True, exist_ok=True)
            (d / nombre).write_bytes(b"jpg")
        pd.DataFrame(filas).to_excel(datos / "EnRevision.xlsx", index=False)

    def test_clasificados_recibe_el_nombre_pelado(self, entorno):
        app, tmp_path, datos, procesado, clasificados = entorno
        self._procesado_con(
            procesado, datos,
            [("MEDU", "048_a.jpg"), ("ZOEA", "099_b.jpg")],
            [{"segment": "a.jpg", "class_pred": "MEDU", "confidence": 0.48},
             {"segment": "b.jpg", "class_pred": "ZOEA", "confidence": 0.99}],
        )

        app(0.95).run()

        assert os.listdir(clasificados / "ZOEA") == ["b.jpg"]
        # a.jpg es dudoso con umbral 0.95: filtrar_dudosos lo movió, sin prefijo
        assert os.listdir(clasificados / "_DUDOSOS" / "MEDU") == ["a.jpg"]

    def test_el_excel_de_clasificados_cruza_con_las_carpetas(self, entorno):
        """Si el prefijo llegara al Excel o al archivo, este cruce fallaría y
        filtrar_dudosos no encontraría las imágenes que tiene que mover."""
        app, tmp_path, datos, procesado, clasificados = entorno
        self._procesado_con(
            procesado, datos, [("MEDU", "048_a.jpg")],
            [{"segment": "a.jpg", "class_pred": "MEDU", "confidence": 0.48}],
        )

        app(0.95).run()

        df = pd.read_excel(datos / "Historico.xlsx")
        assert list(df["segment"]) == ["a.jpg"]
        assert (clasificados / "_DUDOSOS" / "MEDU" / "a.jpg").exists()

    def test_reasentar_no_duplica_la_imagen(self, entorno):
        """La segunda corrida trae otro prefijo para el mismo segmento."""
        app, tmp_path, datos, procesado, clasificados = entorno
        self._procesado_con(
            procesado, datos, [("MEDU", "048_a.jpg")],
            [{"segment": "a.jpg", "class_pred": "MEDU", "confidence": 0.48}],
        )
        app(0.10).run()

        self._procesado_con(
            procesado, datos, [("MEDU", "097_a.jpg")],
            [{"segment": "a.jpg", "class_pred": "MEDU", "confidence": 0.97}],
        )
        app(0.10).run()

        assert os.listdir(clasificados / "MEDU") == ["a.jpg"]
        df = pd.read_excel(datos / "Historico.xlsx")
        assert list(df["segment"]) == ["a.jpg"]

    def test_un_segmento_con_digitos_propios_no_se_recorta(self, entorno):
        app, tmp_path, datos, procesado, clasificados = entorno
        self._procesado_con(
            procesado, datos, [("MEDU", "001_muestra_3.jpg")],
            [{"segment": "001_muestra_3.jpg", "class_pred": "MEDU", "confidence": 0.99}],
        )

        app(0.10).run()

        assert os.listdir(clasificados / "MEDU") == ["001_muestra_3.jpg"]

    def test_mover_a_mano_la_carpeta_actualiza_el_excel(self, entorno):
        """Si el usuario corrige a mano una mala clasificación moviendo el
        archivo a otra carpeta de clase en Procesado/, el Excel asentado
        tiene que reflejar esa clase nueva, no la que predijo el modelo."""
        app, tmp_path, datos, procesado, clasificados = entorno
        # El Excel dice MEDU (lo que predijo el modelo), pero el archivo
        # físico ya está movido a ZOEA (corrección manual del usuario).
        self._procesado_con(
            procesado, datos, [("ZOEA", "048_a.jpg")],
            [{"segment": "a.jpg", "class_pred": "MEDU", "confidence": 0.48}],
        )

        app(0.10).run()

        assert os.listdir(clasificados / "ZOEA") == ["a.jpg"]
        assert not (clasificados / "MEDU").exists()
        df = pd.read_excel(datos / "Historico.xlsx")
        fila = df[df["segment"] == "a.jpg"].iloc[0]
        assert fila["class_pred"] == "ZOEA"

    def test_mover_dentro_de_la_misma_clase_no_pisa_la_confianza(self, entorno):
        """Mover una imagen entre _DUDOSOS/CLASE y CLASE en EnRevision/, sin
        cambiar de clase, NO tiene que alterar su confidence: ese split lo
        recalcula filtrar_dudosos() por confianza (el umbral de asentar puede
        no ser el mismo que el de clasificar — para eso está "Refiltrar").
        El override manual de seguro/dudoso se hace después, en Historico/,
        con el botón "Actualizar" (ver test_bugs.py TestActualizarClasificados)."""
        app, tmp_path, datos, procesado, clasificados = entorno
        d = procesado / "MEDU"
        d.mkdir(parents=True)
        (d / "048_a.jpg").write_bytes(b"jpg")
        pd.DataFrame([
            {"segment": "a.jpg", "class_pred": "MEDU", "confidence": 0.48},
        ]).to_excel(datos / "EnRevision.xlsx", index=False)

        app(0.95).run()

        # confidence sigue siendo la del modelo (0.48): con umbral 0.95 cae
        # en dudosos pese a estar en la carpeta plana de EnRevision/.
        assert os.listdir(clasificados / "_DUDOSOS" / "MEDU") == ["a.jpg"]
        df = pd.read_excel(datos / "Historico.xlsx")
        fila = df[df["segment"] == "a.jpg"].iloc[0]
        assert fila["confidence"] == 0.48

    def test_sigue_funcionando_sin_procesado_excel(self, entorno):
        """Asentar a mano, sin planilla: el prefijo se saca igual."""
        app, tmp_path, datos, procesado, clasificados = entorno
        d = procesado / "MEDU"
        d.mkdir(parents=True)
        (d / "048_a.jpg").write_bytes(b"jpg")

        app(0.10).run()

        assert os.listdir(clasificados / "MEDU") == ["a.jpg"]
