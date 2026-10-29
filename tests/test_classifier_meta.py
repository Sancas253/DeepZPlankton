import json

import pytest

from deepzplankton.core.classifier import REQUIRED_FIELDS, load_meta


def _meta_onnx(**overrides):
    meta = {
        "framework":     "onnx",
        "image_size":    [224, 224],
        "class_names":   ["AMPH", "MEDU", "ZOEA"],
        "preprocessing": "imagenet_meanstd",
    }
    meta.update(overrides)
    return meta


def test_loads_a_valid_onnx_manifest(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps(_meta_onnx()), encoding="utf-8")
    meta = load_meta(str(tmp_path))
    assert meta["framework"] == "onnx"
    assert len(meta["class_names"]) == 3


def test_arch_is_no_longer_required(tmp_path):
    """En ONNX la arquitectura vive en el grafo; el campo no se usa."""
    assert "arch" not in REQUIRED_FIELDS
    (tmp_path / "meta.json").write_text(json.dumps(_meta_onnx()), encoding="utf-8")
    assert "arch" not in load_meta(str(tmp_path))


def test_arch_is_still_accepted_when_present(tmp_path):
    """Los manifests ya exportados lo traen: no deben romperse."""
    (tmp_path / "meta.json").write_text(
        json.dumps(_meta_onnx(arch="swin_tiny")), encoding="utf-8"
    )
    assert load_meta(str(tmp_path))["arch"] == "swin_tiny"


def test_load_meta_rejects_missing_required_fields(tmp_path):
    (tmp_path / "meta.json").write_text(
        json.dumps({"framework": "onnx", "arch": "swin_tiny"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing required fields"):
        load_meta(str(tmp_path))


def test_load_meta_requires_a_manifest(tmp_path):
    """Ya no se autogenera un meta.json para SavedModel de TF."""
    (tmp_path / "saved_model.pb").write_bytes(b"tf")
    with pytest.raises(FileNotFoundError):
        load_meta(str(tmp_path))
