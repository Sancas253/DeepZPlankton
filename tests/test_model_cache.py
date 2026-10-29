"""
Tests del caché de modelos y del reconocimiento de directorios de modelo.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEvictModelCache:
    """
    Reemplazar un modelo copia sobre la misma ruta, y el caché se indexa por
    ruta: sin desalojarlo se seguirían usando los pesos viejos.
    """

    def _cache(self):
        import deepzplankton.core.classifier as classifierExe
        classifierExe._MODEL_CACHE.clear()
        return classifierExe

    def test_evicts_plain_string_key(self):
        from deepzplankton.core.paths import model_key
        ce = self._cache()
        ce._MODEL_CACHE[model_key("C:/models/foo")] = "modelo_viejo"
        ce.evict_model_cache("C:/models/foo")
        assert ce._MODEL_CACHE == {}

    def test_evicts_onnx_tuple_keys_for_both_modes(self):
        """_classify_onnx indexa con (clave, best_fold_only): van las dos."""
        from deepzplankton.core.paths import model_key
        ce = self._cache()
        clave = model_key("C:/models/foo")
        ce._MODEL_CACHE[(clave, True)] = "solo_mejor_fold"
        ce._MODEL_CACHE[(clave, False)] = "ensemble"
        ce.evict_model_cache("C:/models/foo")
        assert ce._MODEL_CACHE == {}

    def test_leaves_other_models_alone(self):
        from deepzplankton.core.paths import model_key
        ce = self._cache()
        otro = model_key("C:/models/bar")
        ce._MODEL_CACHE[(model_key("C:/models/foo"), False)] = "a"
        ce._MODEL_CACHE[(otro, False)] = "b"
        ce.evict_model_cache("C:/models/foo")
        assert list(ce._MODEL_CACHE) == [(otro, False)]

    def test_no_argument_clears_everything(self):
        ce = self._cache()
        ce._MODEL_CACHE["a"] = 1
        ce._MODEL_CACHE[("b", True)] = 2
        ce.evict_model_cache()
        assert ce._MODEL_CACHE == {}

    def test_unknown_path_is_a_noop(self):
        ce = self._cache()
        ce._MODEL_CACHE["a"] = 1
        ce.evict_model_cache("C:/models/no_existe")
        assert ce._MODEL_CACHE == {"a": 1}


class TestIsModelDirFolds:
    """
    average_and_export_onnx.py produce fold_0.onnx … fold_4.onnx, que es el
    layout que carga _load_onnx_sessions(). Antes no contaba como modelo y la
    carpeta se rechazaba al agregarla.
    """

    def test_accepts_fold_ensemble_without_meta(self, tmp_path):
        from deepzplankton.core.paths import is_model_dir
        d = tmp_path / "efficientnetb3_ensemble"
        d.mkdir()
        for i in range(5):
            (d / f"fold_{i}.onnx").write_bytes(b"onnx")
        assert is_model_dir(str(d))

    def test_accepts_single_fold(self, tmp_path):
        from deepzplankton.core.paths import is_model_dir
        d = tmp_path / "modelo"
        d.mkdir()
        (d / "fold_0.onnx").write_bytes(b"onnx")
        assert is_model_dir(str(d))

    def test_rejects_fold_dirs_without_onnx(self, tmp_path):
        """fold_0/ como subdirectorio lo resuelve el escaneo de dos niveles."""
        from deepzplankton.core.paths import is_model_dir
        d = tmp_path / "modelo"
        (d / "fold_0").mkdir(parents=True)
        assert not is_model_dir(str(d))

    def test_rejects_unrelated_onnx_name(self, tmp_path):
        from deepzplankton.core.paths import is_model_dir
        d = tmp_path / "modelo"
        d.mkdir()
        (d / "otra_cosa.onnx").write_bytes(b"onnx")
        assert not is_model_dir(str(d))

    def test_still_accepts_classic_markers(self, tmp_path):
        from deepzplankton.core.paths import is_model_dir
        for marker in ("meta.json", "model.onnx"):
            d = tmp_path / marker.replace(".", "_")
            d.mkdir()
            (d / marker).write_bytes(b"x")
            assert is_model_dir(str(d)), marker

    def test_rejects_leftover_tf_or_torch_markers(self, tmp_path):
        """La app ya sólo ejecuta ONNX: una carpeta con únicamente
        saved_model.pb o model.pt (resabios de TF/torch) no es un modelo
        utilizable y no debe aparecer como tal en el combo de modelos."""
        from deepzplankton.core.paths import is_model_dir
        for marker in ("saved_model.pb", "model.pt"):
            d = tmp_path / marker.replace(".", "_")
            d.mkdir()
            (d / marker).write_bytes(b"x")
            assert not is_model_dir(str(d)), marker
