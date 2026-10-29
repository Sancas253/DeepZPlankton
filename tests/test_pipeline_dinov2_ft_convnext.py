"""dinov2_ft_convnext_tiny sin red: ViT-S/14 DINOv2 en lugar del B, ConvNeXt-Tiny real sin pesos.

python -m unittest discover -s tests -p test_pipeline_dinov2_ft_convnext.py -v
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

PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE))
import deepz_pipeline
import exportar_onnx
import modelo_dinov2_spatial

ARCH = "dinov2_ft_convnext_tiny"
CREATE_MODEL = timm.create_model


def offline_timm(name, **kwargs):
    # El ViT-B real hace lento el test en CPU; el S tiene la misma estructura.
    if name.startswith("vit_base_patch14_dinov2"):
        name = "vit_small_patch14_dinov2"
    kwargs['pretrained'] = False
    return CREATE_MODEL(name, **kwargs)


class DINOv2FTConvNeXtTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.patcher = patch.object(modelo_dinov2_spatial.timm, "create_model",
                                    side_effect=offline_timm)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_shapes(self):
        model = modelo_dinov2_spatial.DINOv2FTConvNeXt(3, image_size=224).eval()
        self.assertEqual(model.n_layers, 4)
        self.assertEqual(model.adapter[1].out_channels, 384)
        with torch.no_grad():
            self.assertEqual(model(torch.randn(2, 3, 224, 224)).shape, (2, 3))
        with self.assertRaises(ValueError):
            modelo_dinov2_spatial.DINOv2FTConvNeXt(3, image_size=210)

    def test_phases_and_optimizer(self):
        from timm.scheduler import CosineLRScheduler

        model = modelo_dinov2_spatial.DINOv2FTConvNeXt(2, image_size=112)
        optimizer = model.make_optimizer(1e-4, 1e-3, 0.05, 0.9, 5e-5, 0.75)
        scheduler = CosineLRScheduler(optimizer, t_initial=30, warmup_t=3, warmup_lr_init=1e-6)
        scheduler.step(3)
        params = [p for g in optimizer.param_groups for p in g['params']]
        self.assertEqual(len(params), len({id(p) for p in params}))
        self.assertEqual({id(p) for p in params}, {id(p) for p in model.parameters()})
        rates = {id(p): g['lr'] for g in optimizer.param_groups for p in g['params']}
        top = rates[id(model.fc.weight)]
        self.assertAlmostEqual(top, max(rates.values()))
        vit_rates = [rates[id(p)] for p in model.vit.parameters()]
        self.assertAlmostEqual(max(vit_rates) / top, 5e-5 / 1e-3)
        self.assertLess(rates[id(model.vit.patch_embed.proj.weight)],
                        rates[id(model.vit.blocks[-1].mlp.fc1.weight)])

        images = torch.randn(2, 3, 112, 112)
        model.set_adapter_only(True)
        model.train()
        self.assertFalse(model.vit.training)
        self.assertFalse(model.classifier.training)
        model(images).sum().backward()
        self.assertGreater(model.adapter[1].weight.grad.abs().sum(), 0)
        self.assertGreater(model.layer_mix.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for p in model.vit.parameters()))
        self.assertTrue(all(p.grad is None for p in model.classifier.parameters()))

        model.zero_grad(set_to_none=True)
        model.set_adapter_only(False)
        model.train()
        self.assertTrue(model.vit.training)
        model(images).sum().backward()
        for module in (model.vit, model.classifier):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in module.parameters()))

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
                self.assertEqual(json.loads(archive.read("meta.json"))["arch"], ARCH)
            session = ort.InferenceSession(
                str(root / "resultado" / f"{ARCH}_ensemble" / "fold_0.onnx"),
                providers=["CPUExecutionProvider"])
            actual = session.run(None, {"input": images.numpy()})[0]
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
            del model, session


if __name__ == "__main__":
    unittest.main()
