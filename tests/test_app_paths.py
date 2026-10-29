import os

from deepzplankton.core.paths import app_dir, app_path, is_model_dir


def test_app_dir_defaults_to_repo_root():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    expected = os.path.join(repo_root, "deepzplankton")
    assert app_dir() == expected


def test_app_path_uses_platform_separators():
    expected = os.path.join(app_dir(), "Segmentos", "_Cache")
    assert app_path("Segmentos", "_Cache") == expected


def test_model_root_is_a_container_not_a_model():
    """
    model/ era el SavedModel de TF que venía con la app, así que la carpeta
    misma daba is_model_dir(). Ahora es sólo el lugar donde se instalan los
    modelos ONNX: cada uno en su subcarpeta.
    """
    raiz = app_path("model")
    if not os.path.isdir(raiz):
        return
    assert not is_model_dir(raiz)
    subdirs = [os.path.join(raiz, d) for d in os.listdir(raiz)]
    subdirs = [d for d in subdirs if os.path.isdir(d)]
    assert subdirs, "model/ debería traer al menos un modelo instalado"
    assert all(is_model_dir(d) for d in subdirs)
