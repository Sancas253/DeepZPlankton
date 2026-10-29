import shutil

from PyQt5.QtCore import QThread, pyqtSignal

from .. import runtime_paths as rt
from ..summary import resumen_procesado


class ClassifierThread(QThread):
    """Clasificación directa: sobre lo que ya haya en el caché, o sobre
    imágenes que el usuario arrastró a mano."""

    finished = pyqtSignal()
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)  # (hecho, total) batches

    def __init__(self, modelo_path, umbral, imagenes=None, best_fold_only=False):
        super().__init__()

        self.modelo_path = modelo_path
        self.umbral = umbral
        self.imagenes = imagenes or []  # images dragged by the user, if any
        self.best_fold_only = best_fold_only

    def run(self):

        try:

            self.log.emit("Inicializando módulos...")

            import os

            from ...core.classifier import ProcesoCancelado, classifier, vaciar_carpeta

            modo = "mejor fold" if self.best_fold_only else "ensemble"
            self.log.emit(f"\n=== CLASIFICACIÓN DIRECTA ===")
            self.log.emit(f"Modelo: {self.modelo_path}")
            self.log.emit(f"Umbral dudosos: {self.umbral}  |  Modo: {modo}")

            if self.imagenes:
                # User dragged images/folder → stage them into the cache
                self.log.emit(f"\n📂 Copiando {len(self.imagenes)} imágenes al caché...")
                os.makedirs(rt.ruta_test, exist_ok=True)  # create first, then clear
                vaciar_carpeta(rt.ruta_test)
                # Copia de verdad, no enlace: estas imágenes son del usuario y
                # están donde él las tenga. Enlazarlas ataría el archivo de la
                # app a un original que puede editar o mover; el resto de las
                # duplicaciones son entre carpetas que la app sí controla.
                for src in self.imagenes:
                    dst = os.path.join(rt.ruta_test, os.path.basename(src))
                    shutil.copy(src, dst)
                self.log.emit("✅ Imágenes copiadas.")
            else:
                self.log.emit("\n📂 Usando imágenes ya presentes en el caché...")

            classifier(self.modelo_path, rt.ruta_test, rt.ruta_procesado, rt.excel_path,
                       best_fold_only=self.best_fold_only, output_excel_path=rt.procesado_excel,
                       on_progress=lambda hecho, total: self.progress.emit(hecho, total),
                       should_stop=self.isInterruptionRequested, umbral=self.umbral)

            resumen = resumen_procesado(self.umbral)
            if resumen:
                self.log.emit(resumen)
            self.log.emit("\n✔ Clasificación finalizada. Revisá y luego presioná Actualizar.")

        except ProcesoCancelado:

            self.log.emit("\n⏹ Cancelado por el usuario.")

        except Exception as e:

            self.log.emit(f"\n❌ Error: {e}")

        self.finished.emit()
