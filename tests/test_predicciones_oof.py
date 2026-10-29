"""Predicciones por imagen, etapa analizar, reconstrucción y comparación de corridas.

python -m unittest discover -s tests -p test_predicciones_oof.py -v
"""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import timm

PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE))
import comparar_corridas
import deepz_pipeline
import predicciones
import reconstruir_predicciones

ARCH = "convnext_dinov2_fusion_plus"
CREATE_MODEL = timm.create_model


def offline_timm(name, **kwargs):
    if name.startswith("vit_base_patch14_dinov2"):
        name = "vit_small_patch14_dinov2"
    kwargs['pretrained'] = False
    return CREATE_MODEL(name, **kwargs)


def leer_csv(ruta):
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class EntrenamientoConPrediccionesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patcher = patch.object(timm, "create_model", side_effect=offline_timm)
        cls.patcher.start()
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.data = root / "dataset"
        for label in range(2):
            folder = cls.data / f"clase_{label}"
            folder.mkdir(parents=True)
            for i in range(6):
                Image.new("RGB", (40, 32), (label * 150, i * 30, 80)).save(folder / f"{i}.png")
        cls.salida = root / ARCH
        with contextlib.redirect_stdout(io.StringIO()):
            deepz_pipeline.etapa_entrenar(
                ARCH, str(cls.data), str(cls.salida),
                {"N_SPLITS": "3", "EPOCHS": "1", "BATCH_SIZE": "2", "IMAGE_SIZE": "56", "VIT_SIZE": "56",
                 "ADAPTER_EPOCHS": "0", "WARMUP_EPOCHS": "1"}, aumentar=False)

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        cls.tmp.cleanup()

    def test_oof_una_fila_por_imagen(self):
        filas = leer_csv(self.salida / "oof_predicciones.csv")
        rutas = sorted(str(p.relative_to(self.data)).replace("\\", "/") for p in self.data.glob("*/*.png"))
        self.assertEqual(sorted(f["ruta"] for f in filas), rutas)
        for f in filas:
            probs = [float(v) for k, v in f.items() if k.startswith("p_")]
            self.assertAlmostEqual(sum(probs), 1.0, places=3)
            self.assertEqual(f["predicha"], f"clase_{int(np.argmax(probs))}")
            self.assertEqual(int(f["acierto"]), int(f["predicha"] == f["real"]))
        # coincide con la matriz OOF de cv_results.json
        cv = json.loads((self.salida / "cv_results.json").read_text(encoding="utf-8"))
        aciertos = sum(int(f["acierto"]) for f in filas)
        self.assertEqual(aciertos, int(np.trace(np.array(cv["pooled_oof"]["confusion_matrix"]))))
        for k in range(3):
            split = json.loads((self.salida / f"fold_{k}/split.json").read_text(encoding="utf-8"))
            test = {f["ruta"] for f in filas if f["fold"] == str(k)}
            self.assertEqual({r.replace("\\", "/") for r in split["test"]}, test)

    def test_etapa_analizar(self):
        with contextlib.redirect_stdout(io.StringIO()):
            deepz_pipeline.etapa_analizar(str(self.salida), str(self.data), max_cam=2)
        for k in range(3):
            carpeta = self.salida / "analisis" / f"fold_{k}"
            filas = leer_csv(carpeta / "predicciones.csv")
            self.assertEqual(len(filas), 4)  # solo el test de ese fold
            self.assertTrue(list((carpeta / "gradcam").rglob("*.png")))

    def test_reconstruir_da_las_mismas_predicciones(self):
        copia = Path(self.tmp.name) / "viejo"
        shutil.copytree(self.salida, copia)
        originales = predicciones.leer_oof(copia)
        for k in range(3):  # como un entrenamiento viejo: sin split ni predicciones
            (copia / f"fold_{k}/split.json").unlink()
            (copia / f"fold_{k}/predicciones_test.csv").unlink()
        (copia / "oof_predicciones.csv").unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            reconstruir_predicciones.main(["--modelo", str(copia), "--dataset", str(self.data),
                                           "--batch", "2"])
        nuevas = predicciones.leer_oof(copia)
        self.assertEqual(set(nuevas), set(originales))
        for ruta, fila in originales.items():
            self.assertEqual(nuevas[ruta]["fold"], fila["fold"])
            self.assertEqual(nuevas[ruta]["predicha"], fila["predicha"])
            self.assertAlmostEqual(float(nuevas[ruta]["confianza"]), float(fila["confianza"]), places=4)

    def test_reconstruir_rechaza_otro_dataset(self):
        otro = Path(self.tmp.name) / "otro_dataset"
        shutil.copytree(self.data, otro)
        (otro / "clase_0" / "0.png").unlink()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            reconstruir_predicciones.main(["--modelo", str(self.salida), "--dataset", str(otro)])


class CompararCorridasTest(unittest.TestCase):
    def escribir(self, carpeta, aciertos, n=200):
        os.makedirs(carpeta, exist_ok=True)
        with open(os.path.join(carpeta, "oof_predicciones.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["fold", "ruta", "real", "predicha", "confianza", "acierto"])
            for i in range(n):
                real = "a" if i % 2 else "b"
                ok = i in aciertos
                w.writerow([0, f"{real}/{i}.png", real, real if ok else ("b" if real == "a" else "a"),
                            0.9, int(ok)])

    def test_mcnemar_y_errores_compartidos(self):
        with tempfile.TemporaryDirectory() as tmp:
            todas = set(range(200))
            dificiles = set(range(10))  # fallan los tres
            self.escribir(f"{tmp}/A", todas - dificiles)
            self.escribir(f"{tmp}/B", todas - dificiles - set(range(10, 40)))  # 30 errores propios
            self.escribir(f"{tmp}/C", todas - dificiles - {50})
            with contextlib.redirect_stdout(io.StringIO()):
                comparar_corridas.main([f"A={tmp}/A", f"B={tmp}/B", f"C={tmp}/C",
                                        "--salida", f"{tmp}/out"])
            pares = {(f["A"], f["B"]): f for f in leer_csv(f"{tmp}/out/pares.csv")}
            ab = pares[("A", "B")]
            self.assertEqual((int(ab["solo_A_acierta"]), int(ab["solo_B_acierta"]),
                              int(ab["ambos_fallan"])), (30, 0, 10))
            self.assertLess(float(ab["p_mcnemar"]), 1e-6)
            self.assertEqual(float(pares[("A", "C")]["p_mcnemar"]), 1.0)  # 0 contra 1
            dif = leer_csv(f"{tmp}/out/dificiles.csv")
            self.assertEqual(sorted(int(Path(f["ruta"]).stem) for f in dif), list(range(10)))
            resumen = Path(f"{tmp}/out/resumen.md").read_text(encoding="utf-8")
            self.assertIn("10 imágenes fallan en los 3 modelos", resumen)
            self.assertTrue(Path(f"{tmp}/out/comparacion.png").exists())

    def test_comparar_con_corridas_anteriores(self):
        with tempfile.TemporaryDirectory() as tmp:
            todas = set(range(200))
            self.escribir(f"{tmp}/nueva_20261014/modelo_x", todas - set(range(5)))
            self.escribir(f"{tmp}/vieja_a_20261001/modelo_a", todas - set(range(8)))
            self.escribir(f"{tmp}/vieja_b_20261002/modelo_b", todas - set(range(3, 12)))
            self.escribir(f"{tmp}/otro_dataset/modelo_c", todas, n=50)
            with open(f"{tmp}/otras_etiquetas/modelo_d/oof_predicciones.csv"
                      if os.makedirs(f"{tmp}/otras_etiquetas/modelo_d") is None else "", "w",
                      newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["fold", "ruta", "real", "predicha", "confianza", "acierto"])
                for i in range(200):
                    real = "a" if i % 2 else "b"
                    w.writerow([0, f"{real}/{i}.png", "z", "z", 0.9, 1])
            salida_entreno = f"{tmp}/nueva_20261014/modelo_x"
            with contextlib.redirect_stdout(io.StringIO()) as log:
                destino = deepz_pipeline.comparar_con_anteriores(salida_entreno, tmp)
            self.assertIn("otro_dataset: otro dataset", log.getvalue())
            self.assertIn("otras_etiquetas: otras etiquetas", log.getvalue())
            nombres = {f["A"] for f in leer_csv(f"{destino}/pares.csv")} | \
                      {f["B"] for f in leer_csv(f"{destino}/pares.csv")}
            self.assertEqual(nombres, {"nueva_20261014 (nueva)", "vieja_a_20261001", "vieja_b_20261002"})
            # sin corridas anteriores no falla
            with tempfile.TemporaryDirectory() as solo:
                self.escribir(f"{solo}/unica/modelo", todas)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertIsNone(deepz_pipeline.comparar_con_anteriores(f"{solo}/unica/modelo", solo))


if __name__ == "__main__":
    unittest.main()
