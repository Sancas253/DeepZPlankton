import os
import shutil

from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox

from ...core import onnx_probe
from ...core.paths import is_model_dir
from .. import runtime_paths as rt
from ..summary import count_procesado
from ..threads.add_model_thread import AgregarModeloThread
from ..widgets.meta_wizard import MetaWizardDialog


class _ModelManagerMixin:
    """Alta, listado y configuración (meta.json) de modelos ONNX instalados."""

    def _ensure_meta(self, model_path):
        """Returns True if meta.json exists or was successfully created."""
        meta_path = os.path.join(model_path, "meta.json")
        if os.path.exists(meta_path):
            return True
        # Sin meta.json sólo se puede armar uno leyendo el grafo ONNX. Un
        # SavedModel de TF o un .pt suelto ya no se ejecutan: hay que
        # convertirlos primero con average_and_export_onnx.py.
        if onnx_probe.describe(model_path) is None:
            self.log_print(
                f"❌ {os.path.basename(model_path)} no contiene un modelo ONNX "
                "(model.onnx o fold_N.onnx) y no tiene meta.json.\n"
                "   Convertilo con average_and_export_onnx.py antes de agregarlo."
            )
            return False
        dlg = MetaWizardDialog(model_path, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self.log_print(f"✅ meta.json guardado para {os.path.basename(model_path)}")
            return True
        return False

    def closeEvent(self, event):
        n = count_procesado()
        if n > 0:
            resp = QMessageBox.question(
                self,
                "Resultados sin asentar",
                f"Hay {n} imágenes en EnRevision/ sin asentar.\n"
                "Los resultados se conservarán al reabrir la app.\n\n"
                "¿Cerrar de todas formas?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if resp == QMessageBox.No:
                event.ignore()
                return
        event.accept()

    def cargar_modelos(self):

        self.combo_modelos.clear()

        # En desarrollo MODEL_BASE y MODEL_BUNDLE son el mismo directorio, así
        # que se descartan repetidos por ruta absoluta.
        vistos = set()

        def _agregar(label, path):
            clave = os.path.normcase(os.path.abspath(path))
            if clave in vistos:
                return
            vistos.add(clave)
            self.combo_modelos.addItem(label, path)

        for base in (rt.MODEL_BASE, rt.MODEL_BUNDLE):
            if not os.path.isdir(base):
                continue

            if is_model_dir(base):
                _agregar(os.path.basename(base), base)

            for d in sorted(os.listdir(base)):
                path = os.path.join(base, d)
                if not os.path.isdir(path):
                    continue
                if is_model_dir(path):
                    _agregar(d, path)
                else:
                    # carpeta padre con folds (ej. swin_tiny/fold_0)
                    for sub in sorted(os.listdir(path)):
                        subpath = os.path.join(path, sub)
                        if is_model_dir(subpath):
                            _agregar(f"{d}/{sub}", subpath)

    def _agregar_desde_carpeta(self):
        carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta del modelo")
        if not carpeta:
            return
        self._lanzar_copia_modelo(carpeta, tmpdir=None)

    def _agregar_desde_zip(self):
        import tempfile
        import zipfile
        ruta_zip, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar ZIP del modelo", "", "Archivos ZIP (*.zip)"
        )
        if not ruta_zip:
            return
        self.log_print(f"📦 Extrayendo {os.path.basename(ruta_zip)}...")
        tmpdir = tempfile.mkdtemp(prefix="deepz_model_")
        try:
            with zipfile.ZipFile(ruta_zip, "r") as zf:
                zf.extractall(tmpdir)
            entries = [e for e in os.listdir(tmpdir) if not e.startswith("__")]
            if len(entries) == 1 and os.path.isdir(os.path.join(tmpdir, entries[0])):
                carpeta = os.path.join(tmpdir, entries[0])
                nombre = None
            else:
                # El ZIP trae los archivos sueltos en la raíz, así que lo que se
                # copia es el directorio temporal: sin esto el modelo quedaba
                # instalado (y listado en el combo) como "deepz_model_ab12cd".
                carpeta = tmpdir
                nombre = os.path.splitext(os.path.basename(ruta_zip))[0]
            self._lanzar_copia_modelo(carpeta, tmpdir=tmpdir, nombre=nombre)
        except Exception as e:
            self.log_print(f"❌ Error extrayendo ZIP: {e}")
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _lanzar_copia_modelo(self, carpeta, tmpdir, nombre=None):
        """Valida la carpeta y lanza AgregarModeloThread para copiar en background."""
        os.makedirs(rt.MODEL_BASE, exist_ok=True)

        # Vale la carpeta del modelo o una carpeta padre con submodelos adentro.
        if not is_model_dir(carpeta) and not any(
            is_model_dir(os.path.join(carpeta, d)) for d in os.listdir(carpeta)
        ):
            self.log_print(
                f"❌ La carpeta no contiene un modelo válido "
                f"(se esperaba meta.json, model.onnx/pt o fold_N.onnx): "
                f"{os.path.basename(carpeta)}"
            )
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
            return
        src = carpeta

        self.set_buttons_enabled(False)
        self.progress_bar.setVisible(True)

        self._agregar_thread = AgregarModeloThread(src, rt.MODEL_BASE, tmpdir=tmpdir,
                                                   nombre=nombre)
        self._agregar_thread.log.connect(self.log_print)
        self._agregar_thread.finished.connect(self._on_modelo_copiado)
        self._agregar_thread.start()

    @staticmethod
    def _modelos_instalados(destino):
        """Los modelos que quedaron adentro de lo que se acaba de copiar.

        Se acepta agregar una carpeta padre con varios modelos adentro, y en ese
        caso el meta.json va en cada modelo: preguntárselo al padre sólo daba el
        error de que no es un ONNX, aunque los hijos estuvieran perfectos.
        """
        if is_model_dir(destino):
            return [destino]
        try:
            hijos = sorted(os.listdir(destino))
        except OSError:
            return []
        return [os.path.join(destino, d) for d in hijos
                if is_model_dir(os.path.join(destino, d))]

    def _on_modelo_copiado(self, destino):
        self.set_buttons_enabled(True)
        self.progress_bar.setVisible(False)
        if destino:
            # El caché de modelos se indexa por ruta y el destino es siempre el
            # mismo para un nombre dado: sin desalojarlo, reemplazar un modelo
            # seguiría usando los pesos viejos hasta reiniciar la app.
            from ...core.classifier import evict_model_cache
            for modelo in self._modelos_instalados(destino):
                evict_model_cache(modelo)
                self._ensure_meta(modelo)
        self.cargar_modelos()
