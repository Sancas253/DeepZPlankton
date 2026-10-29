"""Punto de entrada de la app. Ejecutar como `python -m deepzplankton.main`,
o es el script que apunta DeepZPlankton_V2.spec al empaquetar con PyInstaller.
"""
import ctypes
import os
import sys


def _precargar_dlls_nativas():
    """Precarga DLLs de librerías nativas antes de cualquier otro import.

    Necesario en builds PyInstaller en Windows (WinError 1114): algunas DLLs
    nativas (antes torch, hoy sólo onnxruntime) no se resuelven solas desde
    _internal/ si no se agregan primero al directorio de búsqueda.
    """
    if not hasattr(sys, "_MEIPASS"):
        return
    os.add_dll_directory(sys._MEIPASS)
    for subdir in ["torch/lib", "onnxruntime/capi"]:
        dll_dir = os.path.join(sys._MEIPASS, subdir)
        if not os.path.isdir(dll_dir):
            continue
        os.add_dll_directory(dll_dir)
        for dll in sorted(os.listdir(dll_dir)):
            if dll.endswith(".dll"):
                try:
                    ctypes.WinDLL(os.path.join(dll_dir, dll))
                except Exception:
                    pass


def main():
    _precargar_dlls_nativas()

    # Import absoluto, no relativo: este archivo es el entry point y tanto
    # PyInstaller como `python deepzplankton/main.py` lo ejecutan como
    # __main__, donde un import relativo ("from .gui import ...") falla por
    # no tener un paquete padre conocido.
    if __package__ in (None, ""):
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from PyQt5.QtWidgets import QApplication

    from deepzplankton.gui import runtime_paths
    from deepzplankton.gui.main_window import MainWindow
    from deepzplankton.gui.theme import LIGHT_THEME

    runtime_paths.migrar_nombres_viejos()

    app = QApplication(sys.argv)
    app.setStyleSheet(LIGHT_THEME)

    w = MainWindow()
    w.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
