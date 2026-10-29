from PyQt5.QtWidgets import QApplication, QWidget

from .actions import _ActionsMixin
from .model_manager import _ModelManagerMixin
from .ui_builder import _UiBuilderMixin


class MainWindow(_UiBuilderMixin, _ModelManagerMixin, _ActionsMixin, QWidget):
    """Ventana principal. La construcción de la UI vive en _UiBuilderMixin,
    la gestión de modelos en _ModelManagerMixin y las acciones de los
    botones en _ActionsMixin; acá sólo queda el arranque y los helpers que
    usan las tres partes por igual."""

    def __init__(self):
        super().__init__()
        self._build_ui()

    def log_print(self, text):
        self.log.append(text)
        QApplication.processEvents()

    def _leer_umbral(self):
        """Parsea el campo de umbral y valida que quede en [0, 1].

        El QDoubleValidator del campo ya frena la mayoría de la carga a mano,
        pero no cubre todos los caminos (pegar texto, locale con otro
        separador decimal), así que se revisa también acá antes de arrancar
        cualquier hilo.
        """
        try:
            umbral = float(self.input_umbral.text())
        except ValueError:
            self.log_print("⚠️ Umbral inválido")
            return None
        if not (0.0 <= umbral <= 1.0):
            self.log_print("⚠️ El umbral debe estar entre 0 y 1")
            return None
        return umbral

    def set_buttons_enabled(self, enabled):
        for btn in [self.btn_cargar, self.btn_rapido, self.btn_detalle,
                    self.btn_clasificar, self.btn_actualizar,
                    self.btn_borrar, self.btn_add_model, self.btn_refilter]:
            btn.setEnabled(enabled)
