"""
Tests de onnx_probe: se arma un ModelProto mínimo a mano (sin depender del
paquete onnx ni de los modelos reales, que pesan cientos de MB) y se verifica
que el lector de protobuf saque bien forma de entrada, layout y nº de clases.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deepzplankton.core.onnx_probe as onnx_probe


# ------------------------------------------------------------
# Codificador protobuf mínimo (sólo lo que usa un ModelProto)
# ------------------------------------------------------------
def _varint(n):
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _tag(field, wtype):
    return _varint((field << 3) | wtype)


def _ld(field, payload):
    """Campo length-delimited (wire type 2)."""
    return _tag(field, 2) + _varint(len(payload)) + payload


def _vi(field, value):
    """Campo varint (wire type 0)."""
    return _tag(field, 0) + _varint(value)


def _value_info(name, dims):
    """ValueInfoProto con tensor_type float y las dimensiones dadas."""
    shape = b"".join(
        _ld(1, _vi(1, d) if isinstance(d, int) else _ld(2, d.encode()))
        for d in dims
    )
    tensor = _vi(1, 1) + _ld(2, shape)        # elem_type=FLOAT, shape
    return _ld(1, name.encode()) + _ld(2, _ld(1, tensor))


def _model(input_dims, output_dims, initializer_bytes=0):
    graph = b""
    if initializer_bytes:
        # TensorProto de relleno: verifica que se saltee sin leerlo.
        graph += _ld(5, _ld(9, b"\x00" * initializer_bytes))
    graph += _ld(11, _value_info("input", input_dims))
    graph += _ld(12, _value_info("output", output_dims))
    return _ld(7, graph)


def _write_model(path, input_dims, output_dims, initializer_bytes=0):
    path.write_bytes(_model(input_dims, output_dims, initializer_bytes))


# ------------------------------------------------------------
# probe()
# ------------------------------------------------------------
class TestProbe:

    def test_reads_static_shapes(self, tmp_path):
        f = tmp_path / "model.onnx"
        _write_model(f, [1, 3, 224, 224], [1, 42])
        info = onnx_probe.probe(str(f))
        assert info["input_shape"] == [1, 3, 224, 224]
        assert info["output_shape"] == [1, 42]

    def test_dynamic_batch_dim_is_none(self, tmp_path):
        f = tmp_path / "model.onnx"
        _write_model(f, ["batch_size", 3, 224, 224], ["batch_size", 34])
        info = onnx_probe.probe(str(f))
        assert info["input_shape"] == [None, 3, 224, 224]
        assert info["output_shape"] == [None, 34]

    def test_skips_large_initializers(self, tmp_path):
        """Los pesos se saltean por índice; nunca se copian ni se interpretan."""
        f = tmp_path / "model.onnx"
        _write_model(f, [1, 3, 224, 224], [1, 42], initializer_bytes=2_000_000)
        info = onnx_probe.probe(str(f))
        assert info["input_shape"] == [1, 3, 224, 224]


# ------------------------------------------------------------
# describe()
# ------------------------------------------------------------
class TestDescribe:

    def test_detects_nchw_torch_export(self, tmp_path):
        _write_model(tmp_path / "model.onnx", ["batch", 3, 224, 224], ["batch", 42])
        d = onnx_probe.describe(str(tmp_path))
        assert d["layout"] == "NCHW"
        assert d["image_size"] == [224, 224]
        assert d["n_classes"] == 42

    def test_detects_nhwc_keras_export(self, tmp_path):
        _write_model(tmp_path / "model.onnx", ["unk", 300, 300, 3], ["unk", 34])
        d = onnx_probe.describe(str(tmp_path))
        assert d["layout"] == "NHWC"
        assert d["image_size"] == [300, 300]
        assert d["n_classes"] == 34

    def test_prefers_folds_and_counts_them(self, tmp_path):
        for i in range(5):
            _write_model(tmp_path / f"fold_{i}.onnx", ["b", 3, 224, 224], ["b", 34])
        d = onnx_probe.describe(str(tmp_path))
        assert d["file"] == "fold_0.onnx"
        assert d["n_folds"] == 5

    def test_returns_none_without_onnx(self, tmp_path):
        (tmp_path / "model.pt").write_bytes(b"weights")
        assert onnx_probe.describe(str(tmp_path)) is None

    def test_returns_none_on_unreadable_file(self, tmp_path):
        (tmp_path / "model.onnx").write_bytes(b"no soy un protobuf valido")
        assert onnx_probe.describe(str(tmp_path)) is None

    def test_layout_none_for_unexpected_rank(self, tmp_path):
        _write_model(tmp_path / "model.onnx", ["b", 512], ["b", 34])
        d = onnx_probe.describe(str(tmp_path))
        assert d["layout"] is None
        assert d["image_size"] is None
        assert d["n_classes"] == 34


# ------------------------------------------------------------
# Coherencia con el registro de preprocesado
# ------------------------------------------------------------
class TestLayoutContract:

    def test_every_onnx_preprocessor_declares_a_layout(self):
        from deepzplankton.core.classifier import PREPROCESSORS, PREPROC_LAYOUT
        assert set(PREPROCESSORS) == set(PREPROC_LAYOUT)

    def test_keras_preprocessors_pass_raw_0_255_through(self):
        """El grafo exportado de Keras normaliza adentro: acá no se toca nada."""
        import numpy as np
        from deepzplankton.core.classifier import PREPROCESSORS
        batch = np.full((1, 4, 4, 3), 255.0, dtype=np.float32)
        for name in ("efficientnet", "convnext"):
            out = PREPROCESSORS[name](batch)
            assert out.shape == batch.shape, f"{name} no debe transponer (NHWC)"
            assert out.max() == 255.0, f"{name} no debe reescalar"

    def test_torch_preprocessors_emit_nchw(self):
        import numpy as np
        from deepzplankton.core.classifier import PREPROCESSORS, PREPROC_LAYOUT
        batch = np.zeros((2, 8, 6, 3), dtype=np.float32)
        for name, layout in PREPROC_LAYOUT.items():
            if layout != "NCHW":
                continue
            out = PREPROCESSORS[name](batch)
            assert out.shape == (2, 3, 8, 6), f"{name} debe entregar BCHW"
