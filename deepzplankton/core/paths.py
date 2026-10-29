import os
import sys


APP_NAME = "DeepZPlankton"
MODEL_MARKERS = ("meta.json", "model.onnx")


# Raíz del paquete (deepzplankton/), dos niveles arriba de este archivo
# (deepzplankton/core/paths.py). Es el equivalente, en dev, de "al lado del
# script principal" — que es de donde salían las rutas antes de reorganizar
# el proyecto en paquetes.
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource_path(*parts):
    """Resolve a path inside the PyInstaller bundle (o la raíz del paquete en dev)."""
    base = getattr(sys, "_MEIPASS", _PACKAGE_ROOT)
    return os.path.join(base, *parts)


def app_dir():
    bundled_dir = resource_path(APP_NAME)
    if os.path.isdir(bundled_dir):
        return bundled_dir
    return resource_path()


def app_path(*parts):
    """Path to a bundled resource (model files, etc.). Read-only when frozen."""
    return os.path.join(app_dir(), *parts)


def data_path(*parts):
    """
    Path for user-generated output (Segmentos, Historico, Datos, …).

    When frozen by PyInstaller, resolves to the directory that contains the
    .exe so the user can actually find their data. In development it is the
    same as app_path() (repo root).
    """
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = _PACKAGE_ROOT
    return os.path.join(base, *parts)


def model_path(model_dir):
    """
    Forma canónica del directorio de un modelo.

    Acepta las dos formas que circulan por la app:
      - absoluta, que es lo normal (la arma el combo a partir de MODEL_BASE /
        MODEL_BUNDLE, o la elige el usuario);
      - relativa al bundle, p. ej. "model", como se llamaba antes al modelo
        que venía con el .exe.

    Devuelve siempre una ruta absoluta y normalizada, y es idempotente: aplicarla
    de nuevo sobre su propio resultado no cambia nada. Así el manifest y los
    pesos se resuelven por el mismo camino — antes load_meta() abría
    model_dir/meta.json tal cual mientras la carga de sesiones lo pasaba por
    _MEIPASS, y con una ruta relativa en la app congelada cada uno leía de un
    lugar distinto.
    """
    return os.path.normpath(app_path(model_dir))


def model_key(model_dir):
    """Clave estable para cachear/desalojar un modelo (Windows no distingue
    mayúsculas, así que dos rutas iguales escritas distinto son el mismo modelo)."""
    return os.path.normcase(model_path(model_dir))


def is_model_dir(path):
    if not os.path.isdir(path):
        return False
    if any(os.path.exists(os.path.join(path, marker)) for marker in MODEL_MARKERS):
        return True
    # Ensemble exportado como fold_0.onnx, fold_1.onnx… todavía sin meta.json.
    # Es el layout que produce average_and_export_onnx.py y el que carga
    # _load_onnx_sessions(), así que tiene que contar como modelo válido.
    try:
        return any(
            f.startswith("fold_") and f.endswith(".onnx")
            for f in os.listdir(path)
        )
    except OSError:
        return False
