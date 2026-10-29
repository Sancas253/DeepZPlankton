from PyQt5.QtCore import QThread, pyqtSignal


class RefilterThread(QThread):
    """Reaplica el umbral a toda la base ya clasificada (no sólo lo nuevo)."""

    finished = pyqtSignal()
    log = pyqtSignal(str)

    def __init__(self, umbral):
        super().__init__()
        self.umbral = umbral

    def run(self):

        try:

            self.log.emit(f"=== REFILTRANDO CON UMBRAL {self.umbral} ===")

            from ...core.filtering import refilter_all
            refilter_all(self.umbral)

            self.log.emit("\n✔ Refiltrado finalizado")

        except Exception as e:

            self.log.emit(f"\n❌ Error: {e}")

        self.finished.emit()
