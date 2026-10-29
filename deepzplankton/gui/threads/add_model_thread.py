import os
import shutil

from PyQt5.QtCore import QThread, pyqtSignal


class AgregarModeloThread(QThread):
    """Copia un modelo (carpeta o ZIP ya extraído) a MODEL_BASE en background."""

    finished  = pyqtSignal(str)   # emite la ruta destino copiada, o "" si error
    log       = pyqtSignal(str)

    def __init__(self, src_carpeta, dest_base, tmpdir=None, nombre=None):
        super().__init__()
        self.src_carpeta = src_carpeta
        self.dest_base   = dest_base
        self.tmpdir      = tmpdir   # directorio temporal a limpiar al final (ZIP)
        # Con qué nombre queda instalado. Por defecto el de la carpeta origen,
        # salvo que el llamador sepa uno mejor (ZIP sin carpeta raíz adentro).
        self.nombre      = nombre or os.path.basename(os.path.normpath(src_carpeta))

    def run(self):
        provisorio = None
        try:
            nombre  = self.nombre
            destino = os.path.join(self.dest_base, nombre)
            self.log.emit(f"📋 Copiando {nombre}...")
            # Se copia al lado y recién ahí se reemplaza: borrar el destino
            # primero deja sin modelo si la copia falla a mitad (y son cientos
            # de MB, así que fallar a mitad es una posibilidad real).
            provisorio = destino + ".nuevo"
            shutil.rmtree(provisorio, ignore_errors=True)
            shutil.copytree(self.src_carpeta, provisorio)
            if os.path.exists(destino):
                shutil.rmtree(destino)
            os.replace(provisorio, destino)
            provisorio = None
            self.log.emit(f"✔ Modelo agregado: {nombre}")
            self.finished.emit(destino)
        except Exception as e:
            self.log.emit(f"❌ Error copiando modelo: {e}")
            self.finished.emit("")
        finally:
            if provisorio:
                shutil.rmtree(provisorio, ignore_errors=True)
            if self.tmpdir:
                shutil.rmtree(self.tmpdir, ignore_errors=True)
