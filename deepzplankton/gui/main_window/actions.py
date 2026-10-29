from PyQt5.QtWidgets import QFileDialog, QMessageBox

from ..summary import count_procesado
from ..threads.classification_thread import ClassifierThread
from ..threads.refilter_thread import RefilterThread
from ..threads.segmentation_thread import WorkerThread
from ..threads.settle_thread import ActualizarThread


class _ActionsMixin:
    """Handlers de los botones: cargar imágenes y disparar cada hilo de
    trabajo (segmentar, clasificar, asentar, refiltrar)."""

    def cargar_imagenes(self):

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Seleccionar imágenes",
            "",
            "Imágenes (*.jpg *.jpeg *.png *.tif *.tiff *.bmp)"
        )

        for p in paths:
            self.dropgrid.add_thumbnail(p)

    def cargar_carpeta(self):

        folder = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar carpeta"
        )

        if folder:
            self.dropgrid.add_folder(folder)

    def iniciar_proceso(self, modo):

        imagenes = self.dropgrid.get_all_images()

        if not imagenes:
            self.log_print("⚠️ No hay imágenes cargadas.")
            return

        try:
            min_area = int(self.input_area.text())
        except ValueError:
            self.log_print("⚠️ Área mínima inválida")
            return

        umbral = self._leer_umbral()
        if umbral is None:
            return

        modelo_path = self.combo_modelos.currentData() or ""
        if not modelo_path:
            self.log_print("⚠️ No hay modelos cargados.")
            return

        if not self._ensure_meta(modelo_path):
            self.log_print("⚠️ Se canceló la configuración del modelo.")
            return

        # Advertir si hay resultados sin asentar en EnRevision/
        n_procesado = count_procesado()
        if n_procesado > 0:
            resp = QMessageBox.question(
                self,
                "Resultados sin asentar",
                f"Hay {n_procesado} imágenes en EnRevision/ sin asentar.\n"
                "Si continuás, se descartarán. ¿Continuar de todas formas?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if resp == QMessageBox.No:
                return

        self.set_buttons_enabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)

        self.worker = WorkerThread(
            modo,
            imagenes,
            min_area,
            modelo_path,
            umbral,
            best_fold_only=self.chk_mejor_fold.isChecked(),
        )

        self.worker.log.connect(self.log_print)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.proceso_finalizado)

        self._active_worker = self.worker
        self.btn_cancelar.setEnabled(True)

        self.worker.start()

    def iniciar_clasificacion(self):

        umbral = self._leer_umbral()
        if umbral is None:
            return

        modelo_path = self.combo_modelos.currentData() or ""
        if not modelo_path:
            self.log_print("⚠️ No hay modelos cargados.")
            return

        if not self._ensure_meta(modelo_path):
            self.log_print("⚠️ Se canceló la configuración del modelo.")
            return

        # Collect whatever the user dragged in. If the grid is empty the
        # thread will classify whatever is already in the cache (a previous
        # segmentation run), which is the intended re-classify use-case.
        imagenes = self.dropgrid.get_all_images()

        self.set_buttons_enabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)

        self.classifier_worker = ClassifierThread(
            modelo_path, umbral, imagenes,
            best_fold_only=self.chk_mejor_fold.isChecked(),
        )

        self.classifier_worker.log.connect(self.log_print)
        self.classifier_worker.progress.connect(self.on_progress)
        self.classifier_worker.finished.connect(self.proceso_finalizado)

        self._active_worker = self.classifier_worker
        self.btn_cancelar.setEnabled(True)

        self.classifier_worker.start()

    def iniciar_actualizar(self):
        umbral = self._leer_umbral()
        if umbral is None:
            return

        self.set_buttons_enabled(False)
        self.progress_bar.setVisible(True)

        self.actualizar_worker = ActualizarThread(umbral)
        self.actualizar_worker.log.connect(self.log_print)
        self.actualizar_worker.finished.connect(self.proceso_finalizado)
        self.actualizar_worker.start()

    def iniciar_refilter(self):

        umbral = self._leer_umbral()
        if umbral is None:
            return

        self.set_buttons_enabled(False)
        self.progress_bar.setVisible(True)

        self.refilter_worker = RefilterThread(umbral)

        self.refilter_worker.log.connect(self.log_print)
        self.refilter_worker.finished.connect(self.proceso_finalizado)

        self.refilter_worker.start()

    def proceso_finalizado(self):

        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # vuelve a indeterminada para la próxima corrida
        self.set_buttons_enabled(True)
        self.btn_cancelar.setEnabled(False)
        self._active_worker = None
        self.log_print("\nListo para nueva ejecución\n")

    def on_progress(self, hecho, total):
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(hecho)

    def cancelar_proceso(self):
        if self._active_worker is None:
            return
        self._active_worker.requestInterruption()
        self.btn_cancelar.setEnabled(False)  # evitar doble clic mientras termina el batch en curso
        self.log_print("\n⏳ Cancelando... termina el batch en curso y corta ahí.")
