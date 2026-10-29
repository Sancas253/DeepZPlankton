import os
import shutil

from PyQt5.QtCore import QThread, pyqtSignal

from ...core.classifier import ProcesoCancelado, classifier
from ...segmentation.fast import Segmenter_No_View
from ...segmentation.interactive import Segmenter_With_View
from .. import runtime_paths as rt
from ..summary import resumen_procesado


class WorkerThread(QThread):
    """Segmentación (rápida o con vista) seguida de clasificación."""

    finished = pyqtSignal()
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, modo, imagenes, min_area, modelo_path, umbral, best_fold_only=False):
        super().__init__()

        self.modo = modo
        self.imagenes = imagenes
        self.min_area = min_area
        self.modelo_path = modelo_path
        self.umbral = umbral
        self.best_fold_only = best_fold_only

    def run(self):

        try:

            self.log.emit("Inicializando módulos...")

            self.log.emit(f"\n=== SEGMENTACIÓN {self.modo.upper()} ===")
            self.log.emit(f"Área mínima: {self.min_area}")
            self.log.emit(f"Modelo: {self.modelo_path}")
            self.log.emit(f"Umbral dudosos: {self.umbral}")
            self.log.emit(f"Cantidad de imágenes: {len(self.imagenes)}")

            # Limpiar cache y EnRevision antes de la nueva segmentación
            if os.path.exists(rt.ruta_test):
                shutil.rmtree(rt.ruta_test)
            os.makedirs(rt.ruta_test, exist_ok=True)
            if os.path.exists(rt.ruta_procesado):
                shutil.rmtree(rt.ruta_procesado)
            os.makedirs(rt.ruta_procesado, exist_ok=True)
            if os.path.exists(rt.procesado_excel):
                os.remove(rt.procesado_excel)

            should_stop = self.isInterruptionRequested
            on_progress = lambda hecho, total: self.progress.emit(hecho, total)

            if self.modo == "rapido":
                # Segmenter_With_View (modo "detalle") ya tiene su propio
                # corte interactivo por tecla (ver cv2.waitKey ahí); acá se
                # cablea sólo para el modo rápido, que es el que corre
                # desatendido sobre campañas grandes.
                Segmenter_No_View(self.imagenes, self.min_area,
                                   on_progress=on_progress, should_stop=should_stop)
            else:
                Segmenter_With_View(self.imagenes, self.min_area)

            classifier(self.modelo_path, rt.ruta_test, rt.ruta_procesado, rt.excel_path,
                       best_fold_only=self.best_fold_only, output_excel_path=rt.procesado_excel,
                       on_progress=on_progress, should_stop=should_stop, umbral=self.umbral)

            resumen = resumen_procesado(self.umbral)
            if resumen:
                self.log.emit(resumen)
            self.log.emit("\n✔ Clasificación finalizada. Revisá y luego presioná Actualizar.")

        except ProcesoCancelado:

            self.log.emit("\n⏹ Cancelado por el usuario.")

        except Exception as e:

            self.log.emit(f"\n❌ Error: {e}")

        self.finished.emit()
