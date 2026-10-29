"""dinov2_convnext_tiny_plus sin red: DINO reducido, ConvNeXt-Tiny real sin pesos.

python -m unittest discover -s tests -p test_pipeline_dinov2_convnext_plus.py -v
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch
import timm
from transformers import Dinov2Config, Dinov2Model

PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE))
import deepz_pipeline
import exportar_onnx
import modelo_dinov2_spatial

ARCH = "dinov2_convnext_tiny_plus"
CREATE_MODEL = timm.create_model


def offline_timm(name, **kwargs):
    kwargs['pretrained'] = False
    return CREATE_MODEL(name, **kwargs)


def small_config(**kwargs):
    config = Dinov2Config(hidden_size=24, num_hidden_layers=5, num_attention_heads=3,
                          image_size=518)
    config._attn_implementation = "eager"
    return config


class ConvNeXtPlusTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.object(
            modelo_dinov2_spatial.Dinov2Model, "from_pretrained",
            side_effect=lambda *a, **kw: Dinov2Model(small_config())))
        self.stack.enter_context(patch.object(modelo_dinov2_spatial.timm, "create_model",
                                              side_effect=offline_timm))
        self.stack.enter_context(patch.object(modelo_dinov2_spatial, "Dinov2Config",
                                              side_effect=small_config))

    def tearDown(self):
        self.stack.close()

    def test_features_enter_convnext_stage2(self):
        model = modelo_dinov2_spatial.DINOv2ConvNeXtPlus(3, image_size=280).eval()
        self.assertEqual(model.grid, 20)
        self.assertEqual(model.n_layers, 4)
        self.assertEqual(model.adapter[1].out_channels, 384)
        self.assertEqual(model.fc.in_features, 768 + 24)
        with torch.no_grad():
            logits = model(torch.randn(2, 3, 280, 280))
        self.assertEqual(logits.shape, (2, 3))
        with self.assertRaises(ValueError):
            modelo_dinov2_spatial.DINOv2ConvNeXtPlus(3, image_size=288)

    def test_adapter_warmup_and_optimizer(self):
        from timm.scheduler import CosineLRScheduler

        model = modelo_dinov2_spatial.DINOv2ConvNeXtPlus(2, image_size=224)
        optimizer = model.make_optimizer(1e-4, 1e-3, 0.05, 0.9)
        scheduler = CosineLRScheduler(optimizer, t_initial=30, warmup_t=3, warmup_lr_init=1e-6)
        scheduler.step(3)
        params = [p for g in optimizer.param_groups for p in g['params']]
        self.assertEqual(len(params), len({id(p) for p in params}))
        self.assertEqual({id(p) for p in params},
                         {id(p) for p in model.parameters() if p.requires_grad})
        rates = {id(p): g['lr'] for g in optimizer.param_groups for p in g['params']}
        for p in (model.adapter[1].weight, model.fc.weight, model.layer_mix):
            self.assertAlmostEqual(rates[id(p)], max(rates.values()))

        images = torch.randn(2, 3, 224, 224)
        model.set_adapter_only(True)
        model.train()
        self.assertFalse(model.classifier.training)
        model(images).sum().backward()
        for p in (model.adapter[1].weight, model.fc.weight, model.layer_mix):
            self.assertGreater(p.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for p in model.classifier.parameters()))

        model.zero_grad(set_to_none=True)
        model.set_adapter_only(False)
        model.train()
        self.assertTrue(model.classifier.training)
        model(images).sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.classifier.parameters()))
        self.assertTrue(all(p.grad is None for p in model.dinov2.parameters()))
        self.assertFalse(model.dinov2.training)

    def test_training_matrix_export_zip(self):
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "dataset"
            for label in range(2):
                folder = data / f"clase_{label}"
                folder.mkdir(parents=True)
                for i in range(3):
                    Image.new("RGB", (32, 40), (label * 150, i * 50, 80)).save(folder / f"{i}.png")
            output = root / ARCH
            with contextlib.redirect_stdout(io.StringIO()):
                deepz_pipeline.etapa_entrenar(
                    ARCH, str(data), str(output),
                    {"N_SPLITS": "3", "EPOCHS": "2", "BATCH_SIZE": "2", "IMAGE_SIZE": "112",
                     "ADAPTER_EPOCHS": "1", "WARMUP_EPOCHS": "1"}, aumentar=False)
                summary = json.loads((output / "cv_results.json").read_text(encoding="utf-8"))
                self.assertEqual(len(summary["folds"]), 3)
                self.assertEqual(summary["pooled_oof"]["n_total"], 6)
                meta = json.loads((output / "fold_0/meta.json").read_text(encoding="utf-8"))
                self.assertEqual(meta["arch"], ARCH)
                self.assertEqual(meta["image_size"], [112, 112])

                model = exportar_onnx.build_model(ARCH, 2, meta["image_size"])
                model.load_state_dict(torch.load(output / "fold_0/model.pt", weights_only=True))
                model.eval()
                images = torch.randn(2, 3, 112, 112)
                with torch.no_grad():
                    expected = model(images).numpy()
                deepz_pipeline.etapa_matriz(str(output), ARCH)
                deepz_pipeline.etapa_exportar(str(output), f"{ARCH}_ensemble",
                                              str(root / "resultado"), "")
            with zipfile.ZipFile(root / "resultado" / f"{ARCH}_ensemble.zip") as archive:
                meta = json.loads(archive.read("meta.json"))
                self.assertEqual(meta["framework"], "onnx")
                self.assertEqual(meta["arch"], ARCH)
            session = ort.InferenceSession(
                str(root / "resultado" / f"{ARCH}_ensemble" / "fold_0.onnx"),
                providers=["CPUExecutionProvider"])
            actual = session.run(None, {"input": images.numpy()})[0]
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
            del model, session


if __name__ == "__main__":
    unittest.main()
