"""
Tests de la duplicación de segmentos entre carpetas (archivos.py).

El mismo recorte vive en hasta cuatro lugares. Se duplica con enlaces duros para
no escribir los bytes cuatro veces, y eso es aceptable sólo si se cumplen dos
condiciones, que es lo que se prueba acá:

  1. Cada carpeta se puede borrar por separado. Borrar un enlace —o el original—
     no puede llevarse puestos a los demás.
  2. Nada se escribe encima. El contenido sí es compartido, así que reemplazar
     un archivo tiene que crear una entrada nueva, no modificar la que otras
     carpetas creen suya.

Y cuando el enlace no es posible (otro volumen, red, FAT32), tiene que caer a una
copia de verdad sin que se note.
"""
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepzplankton.core.files import enlazar_o_copiar, liberar_destino

CONTENIDO = b"CROP" * 500


@pytest.fixture
def cuatro_carpetas(tmp_path):
    """Reproduce el recorrido real: Segmentos → _Cache → Procesado → Clasificados."""
    segmentos = tmp_path / "Segmentos" / "muestra"
    segmentos.mkdir(parents=True)
    original = segmentos / "x_0.jpg"
    original.write_bytes(CONTENIDO)

    destinos = []
    for carpeta in ("Segmentos/_Cache", "Procesado/MEDU", "Clasificados/MEDU"):
        d = tmp_path / carpeta
        d.mkdir(parents=True)
        dst = d / "x_0.jpg"
        enlazar_o_copiar(str(original), str(dst))
        destinos.append(dst)
    return original, destinos


def _vivos(paths):
    return [p for p in paths if p.exists()]


def _todos_intactos(paths):
    return all(p.read_bytes() == CONTENIDO for p in _vivos(paths))


class TestBorrarUnoNoBorraLosDemas:
    """La condición innegociable."""

    def test_borrar_procesado_no_toca_al_resto(self, cuatro_carpetas):
        original, destinos = cuatro_carpetas
        cache, procesado, clasificados = destinos

        procesado.unlink()

        assert original.exists() and cache.exists() and clasificados.exists()
        assert _todos_intactos([original] + destinos)

    def test_borrar_el_original_no_rompe_las_copias(self, cuatro_carpetas):
        """Alguien limpia Segmentos/ para hacer lugar: Clasificados/ sigue entero."""
        original, destinos = cuatro_carpetas

        original.unlink()

        assert len(_vivos(destinos)) == 3
        assert _todos_intactos(destinos)

    def test_borrar_el_primero_no_lo_convierte_en_el_dueño(self, cuatro_carpetas):
        """No hay original ni jerarquía: el de Segmentos/ es un nombre más.

        Se puede borrar el primero y seguir trabajando con los otros tres, e
        incluso volver a enlazar desde cualquiera de ellos.
        """
        original, destinos = cuatro_carpetas

        original.unlink()

        assert _todos_intactos(destinos)
        rehecho = original.parent / "recuperado.jpg"
        os.link(str(destinos[-1]), str(rehecho))
        assert rehecho.read_bytes() == CONTENIDO

    def test_el_contenido_vive_hasta_que_se_va_el_ultimo(self, cuatro_carpetas):
        original, destinos = cuatro_carpetas
        original.unlink()
        for d in destinos[:-1]:
            d.unlink()

        ultimo = destinos[-1]
        assert ultimo.read_bytes() == CONTENIDO
        assert os.stat(ultimo).st_nlink == 1

        ultimo.unlink()
        assert not ultimo.exists()

    def test_borrar_tres_deja_la_cuarta_legible(self, cuatro_carpetas):
        original, destinos = cuatro_carpetas
        cache, procesado, clasificados = destinos

        original.unlink()
        cache.unlink()
        procesado.unlink()

        assert clasificados.read_bytes() == CONTENIDO

    def test_borrar_la_carpeta_entera_no_toca_al_resto(self, cuatro_carpetas, tmp_path):
        """Asentar hace exactamente esto: shutil.rmtree(Procesado/)."""
        original, destinos = cuatro_carpetas
        cache, procesado, clasificados = destinos

        shutil.rmtree(tmp_path / "Procesado")

        assert not procesado.exists()
        assert original.exists() and cache.exists() and clasificados.exists()
        assert _todos_intactos([original, cache, clasificados])

    def test_mandar_uno_a_la_papelera_no_toca_al_resto(self, cuatro_carpetas, tmp_path):
        """Borrar desde el explorador es mover a $Recycle.Bin, o sea un rename."""
        original, destinos = cuatro_carpetas
        cache, procesado, clasificados = destinos
        papelera = tmp_path / "Papelera"
        papelera.mkdir()

        shutil.move(str(procesado), str(papelera / "x_0.jpg"))

        assert original.exists() and cache.exists() and clasificados.exists()
        assert _todos_intactos([original, cache, clasificados])
        assert (papelera / "x_0.jpg").read_bytes() == CONTENIDO

    def test_mover_dentro_de_clasificados_no_rompe_nada(self, cuatro_carpetas, tmp_path):
        """filtrar_dudosos mueve a _DUDOSOS/CLASE con shutil.move."""
        original, destinos = cuatro_carpetas
        _, _, clasificados = destinos
        dudosos = tmp_path / "Clasificados" / "_DUDOSOS" / "MEDU"
        dudosos.mkdir(parents=True)

        shutil.move(str(clasificados), str(dudosos / "x_0.jpg"))

        assert (dudosos / "x_0.jpg").read_bytes() == CONTENIDO
        assert original.read_bytes() == CONTENIDO


class TestNoSeEscribeEncima:
    """El contenido es compartido: reemplazar tiene que desvincular primero."""

    def test_reemplazar_un_destino_no_cambia_a_los_demas(self, cuatro_carpetas, tmp_path):
        original, destinos = cuatro_carpetas
        _, procesado, clasificados = destinos

        otro = tmp_path / "otro.jpg"
        otro.write_bytes(b"OTRO CROP DISTINTO")
        enlazar_o_copiar(str(otro), str(procesado))

        assert procesado.read_bytes() == b"OTRO CROP DISTINTO"
        assert original.read_bytes() == CONTENIDO
        assert clasificados.read_bytes() == CONTENIDO

    def test_resegmentar_no_le_cambia_los_pixeles_a_lo_asentado(self, cuatro_carpetas):
        """El caso peligroso: mismo nombre, otro organismo.

        Los segmentos se llaman {muestra}_{n}.jpg por posición, así que
        re-segmentar con otro min_area reusa los nombres con otro contenido.
        Por eso el segmentador desvincula antes de cv2.imwrite: sin eso, una
        imagen ya revisada y asentada cambiaría de píxeles en silencio,
        conservando su nombre y su fila en el Excel.
        """
        original, destinos = cuatro_carpetas
        _, _, clasificados = destinos

        liberar_destino(str(original))          # lo que hace el segmentador
        original.write_bytes(b"OTRO ORGANISMO")

        assert original.read_bytes() == b"OTRO ORGANISMO"
        assert clasificados.read_bytes() == CONTENIDO

    def test_sin_liberar_el_destino_el_contenido_se_propaga(self, cuatro_carpetas):
        """Documenta POR QUÉ hace falta liberar_destino: así se rompería."""
        original, destinos = cuatro_carpetas
        _, _, clasificados = destinos

        original.write_bytes(b"PISADO EN EL LUGAR")   # sin desvincular

        assert clasificados.read_bytes() == b"PISADO EN EL LUGAR"

    def test_liberar_destino_es_inofensivo_si_no_existe(self, tmp_path):
        liberar_destino(str(tmp_path / "no_existe.jpg"))   # no debe romper


class TestComparteElContenido:

    def test_no_vuelve_a_escribir_los_bytes(self, cuatro_carpetas):
        """Si no comparten contenido, no se ahorró nada."""
        original, destinos = cuatro_carpetas
        assert os.stat(original).st_nlink == 4

    def test_el_conteo_baja_al_borrar(self, cuatro_carpetas):
        original, destinos = cuatro_carpetas
        destinos[1].unlink()
        assert os.stat(original).st_nlink == 3


class TestFallbackACopia:
    """Otro volumen, carpeta de red, pendrive FAT32."""

    @pytest.fixture
    def sin_enlaces(self, monkeypatch):
        def _no_se_puede(src, dst):
            raise OSError(1, "Incorrect function")
        monkeypatch.setattr(os, "link", _no_se_puede)

    def test_copia_cuando_no_se_puede_enlazar(self, tmp_path, sin_enlaces):
        src = tmp_path / "x.jpg"
        src.write_bytes(CONTENIDO)
        dst = tmp_path / "destino" / "x.jpg"
        dst.parent.mkdir()

        enlazar_o_copiar(str(src), str(dst))

        assert dst.read_bytes() == CONTENIDO
        assert os.stat(dst).st_nlink == 1        # es una copia independiente

    def test_la_copia_tambien_sobrevive_al_borrado_del_original(self, tmp_path, sin_enlaces):
        src = tmp_path / "x.jpg"
        src.write_bytes(CONTENIDO)
        dst = tmp_path / "x_copia.jpg"

        enlazar_o_copiar(str(src), str(dst))
        src.unlink()

        assert dst.read_bytes() == CONTENIDO

    def test_reemplaza_igual_si_el_destino_existe(self, tmp_path, sin_enlaces):
        src = tmp_path / "x.jpg"
        src.write_bytes(b"NUEVO")
        dst = tmp_path / "y.jpg"
        dst.write_bytes(b"VIEJO")

        enlazar_o_copiar(str(src), str(dst))

        assert dst.read_bytes() == b"NUEVO"
