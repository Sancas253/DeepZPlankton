from PyQt5.QtCore import QObject, pyqtSignal


class EmittingStream(QObject):
    """Redirige stdout/stderr al log de la ventana (ver MainWindow.__init__)."""

    text_written = pyqtSignal(str)

    def write(self, text):
        if text.strip():
            self.text_written.emit(text)

    def flush(self):
        pass
