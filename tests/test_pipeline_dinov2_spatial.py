"""Integración sin red; DINO reducido, clasificadores Tiny reales.

python -m unittest discover -s tests -p test_pipeline_dinov2_spatial.py -v
"""
import ast
import base64
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

CREATE_MODEL = timm.create_model


def offline_timm(name, **kwargs):
    kwargs['pretrained'] = False
    return CREATE_MODEL(name, **kwargs)


def small_config(**kwargs):
    config = Dinov2Config(hidden_size=24, num_hidden_layers=1, num_attention_heads=3,
                          image_size=518)
    config._attn_implementation = "eager"
    return config


class SpatialPipelineTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_five_folds_match_original_training(self):
        from sklearn.model_selection import StratifiedKFold

        def splits_from_script(name):
            tree = ast.parse((PIPELINE / "entrenamiento" / name).read_text(encoding='utf-8'))
            env = dict(np=np, StratifiedKFold=StratifiedKFold, N_SPLITS=5, SEED=42,
                       all_labels=np.repeat(np.arange(3), 10), all_indices=np.arange(30))
            assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                           and isinstance(n.targets[0], ast.Name)
                           and n.targets[0].id in ('skf', 'fold_test_indices')]
            loop = next(n for n in tree.body if isinstance(n, ast.For)
                        and isinstance(n.target, ast.Name) and n.target.id == 'fold_idx')
            partition_nodes = [n for n in loop.body if isinstance(n, ast.Assign)
                               and isinstance(n.targets[0], ast.Name)
                               and n.targets[0].id in ('test_fold', 'val_fold', 'train_folds',
                                                      'test_indices', 'val_indices', 'train_indices')]
            exec(compile(ast.Module(body=assignments, type_ignores=[]), name, 'exec'), env)
            result = []
            for k in range(5):
                env['fold_idx'] = k
                exec(compile(ast.Module(body=partition_nodes, type_ignores=[]), name, 'exec'), env)
                result.append(tuple(env[key].tolist() for key in ('train_indices', 'val_indices', 'test_indices')))
            return result

        expected = splits_from_script('train_dinov2_ft.py')
        for name in ('train_dinov2_convnext_tiny.py', 'train_dinov2_swin_tiny.py'):
            self.assertEqual(splits_from_script(name), expected)
        self.assertEqual(sorted(i for _, _, test in expected for i in test), list(range(30)))

    def test_full_base_architectures_without_download(self):
        for arch in ("dinov2_convnext_tiny", "dinov2_swin_tiny"):
            with self.subTest(arch=arch):
                model = exportar_onnx.build_model(arch + "_transfer", 3, [224, 224]).eval()
                self.assertEqual(model.dinov2.config.hidden_size, 768)
                self.assertEqual(model.dinov2.embeddings.position_embeddings.shape[1], 1370)
                with torch.no_grad():
                    logits = model(torch.randn(1, 3, 224, 224))
                self.assertEqual(logits.shape, (1, 3))
                self.assertTrue(torch.isfinite(logits).all())
                del model

    def test_adapter_warmup_and_optimizer(self):
        from timm.scheduler import CosineLRScheduler

        for backbone in ("convnext_tiny", "swin_tiny"):
            with self.subTest(backbone=backbone), \
                 patch.object(modelo_dinov2_spatial, "Dinov2Config", side_effect=small_config):
                model = modelo_dinov2_spatial.DINOv2Spatial(2, backbone, pretrained=False)
                optimizer = model.make_optimizer(1e-4, 1e-3, 0.05, 0.8)
                scheduler = CosineLRScheduler(optimizer, t_initial=30, warmup_t=3, warmup_lr_init=1e-6)
                scheduler.step(3)
                params = [p for g in optimizer.param_groups for p in g['params']]
                self.assertEqual(len(params), len({id(p) for p in params}))
                expected_ids = {id(p) for p in model.parameters() if p.requires_grad}
                self.assertEqual({id(p) for p in params}, expected_ids)
                rates = {id(p): g['lr'] for g in optimizer.param_groups for p in g['params']}
                self.assertAlmostEqual(rates[id(model.adapter.weight)], max(rates.values()))
                model.set_adapter_only(True)
                model.train()
                self.assertFalse(model.classifier.training)
                images = torch.randn(2, 3, 224, 224)
                model(images).sum().backward()
                self.assertGreater(model.adapter.weight.grad.abs().sum(), 0)
                head_ids = {id(p) for p in model.classifier.get_classifier().parameters()}
                self.assertTrue(all(p.grad is None for p in model.classifier.parameters() if id(p) not in head_ids))
                model.zero_grad(set_to_none=True)
                model.set_adapter_only(False)
                model.train()
                model(images).sum().backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in model.classifier.parameters() if id(p) not in head_ids))
                self.assertTrue(all(p.grad is None for p in model.dinov2.parameters()))
                del model, optimizer, scheduler

    def test_notebook_contains_current_pipeline(self):
        notebook = json.loads((PIPELINE / "Pipeline_DeepZ.ipynb").read_text(encoding="utf-8"))
        cell = "".join(notebook["cells"][1]["source"])
        tree = ast.parse(cell)
        encoded = next(ast.literal_eval(n.value) for n in tree.body
                       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                       and n.targets[0].id == "PIPELINE_ZIP_B64")
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
            for name in archive.namelist():
                self.assertEqual(archive.read(name), (PIPELINE / name).read_bytes(), name)
            self.assertIn("modelo_dinov2_spatial.py", archive.namelist())
            for arch in ("dinov2_convnext_tiny", "dinov2_swin_tiny"):
                self.assertIn(f'"{arch}"', cell.splitlines()[1])
                self.assertIn(f"entrenamiento/train_{arch}.py", archive.namelist())

    def test_training_matrix_export_zip_and_gradients(self):
        import onnxruntime as ort

        for arch in ("dinov2_convnext_tiny", "dinov2_swin_tiny"):
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                data = root / "dataset"
                for label in range(2):
                    folder = data / f"clase_{label}"
                    folder.mkdir(parents=True)
                    for i in range(3):
                        Image.new("RGB", (32, 40), (label * 150, i * 50, 80)).save(folder / f"{i}.png")
                output = root / arch
                with patch.object(modelo_dinov2_spatial.Dinov2Model, "from_pretrained",
                                  side_effect=lambda *a, **kw: Dinov2Model(small_config())), \
                     patch.object(modelo_dinov2_spatial.timm, "create_model", side_effect=offline_timm), \
                     patch.object(modelo_dinov2_spatial, "Dinov2Config", side_effect=small_config), \
                     contextlib.redirect_stdout(io.StringIO()):
                    deepz_pipeline.etapa_entrenar(
                        arch, str(data), str(output),
                        {"N_SPLITS": "3", "EPOCHS": "2", "BATCH_SIZE": "2",
                         "ADAPTER_EPOCHS": "1", "WARMUP_EPOCHS": "1"}, aumentar=False)
                    summary = json.loads((output / "cv_results.json").read_text(encoding="utf-8"))
                    self.assertEqual(len(summary["folds"]), 3)
                    self.assertEqual(summary["pooled_oof"]["n_total"], 6)
                    self.assertEqual(np.asarray(summary["pooled_oof"]["confusion_matrix"]).sum(), 6)
                    self.assertFalse(json.loads((output / "aumentacion.json").read_text())["activa"])
                    from sklearn.model_selection import StratifiedKFold
                    samples = sorted(str(p.relative_to(data)) for p in data.glob('*/*.png'))
                    labels = np.repeat([0, 1], 3)
                    partitions = [test for _, test in StratifiedKFold(3, shuffle=True, random_state=42).split(samples, labels)]
                    for k in range(3):
                        split = json.loads((output / f'fold_{k}/split.json').read_text())
                        self.assertEqual(split['test'], [samples[i] for i in partitions[k]])
                        self.assertEqual(split['val'], [samples[i] for i in partitions[(k + 1) % 3]])
                        self.assertFalse(set(split['train']) & (set(split['val']) | set(split['test'])))
                        self.assertEqual(set(split['train']) | set(split['val']) | set(split['test']), set(samples))
                    model = exportar_onnx.build_model(arch + "_transfer", 2, [224, 224])
                    model.load_state_dict(torch.load(output / "fold_0/model.pt", weights_only=True))
                    model.train()
                    self.assertFalse(model.dinov2.training)
                    images = torch.randn(2, 3, 224, 224)
                    torch.nn.CrossEntropyLoss()(model(images), torch.tensor([0, 1])).backward()
                    self.assertTrue(all(p.grad is None for p in model.dinov2.parameters()))
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                        for p in model.classifier.parameters()))
                    model.eval()
                    with torch.no_grad():
                        expected = model(images).numpy()
                    deepz_pipeline.etapa_matriz(str(output), arch)
                    deepz_pipeline.etapa_exportar(str(output), f"{arch}_ensemble", str(root / "resultado"), "")
                zip_path = root / "resultado" / f"{arch}_ensemble.zip"
                self.assertTrue(zip_path.exists())
                with zipfile.ZipFile(zip_path) as archive:
                    meta = json.loads(archive.read("meta.json"))
                    self.assertEqual(meta["framework"], "onnx")
                    self.assertEqual(meta["n_folds"], 3)
                    self.assertEqual(meta["arch"], arch + "_transfer")
                    self.assertIn("cv_results.json", archive.namelist())
                session = ort.InferenceSession(
                    str(root / "resultado" / f"{arch}_ensemble" / "fold_0.onnx"),
                    providers=["CPUExecutionProvider"])
                actual = session.run(None, {"input": images.numpy()})[0]
                self.assertEqual(actual.shape, (2, 2))
                np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
                del model, session


if __name__ == "__main__":
    unittest.main()
