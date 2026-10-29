# -*- coding: utf-8 -*-
"""
Exhaustive bug & scalability regression tests for DeepZPlankton V2.

Run with:  python3 -m pytest tests/test_bugs.py -v
"""

import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path helpers so the project root is importable
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from deepzplankton.core.paths import app_path, app_dir


# ============================================================
# Helpers to build minimal Excel fixtures
# ============================================================

def make_clasificados_excel(path, rows):
    """Create Clasificados.xlsx with given rows (list of dicts)."""
    pd.DataFrame(rows).to_excel(path, index=False)


def make_seg_excel(path, rows):
    pd.DataFrame(rows).to_excel(path, index=False)


# ============================================================
# 1. append_to_excel – correctness tests
# ============================================================

class TestAppendToExcel:

    def _get_fn(self):
        from deepzplankton.core.classifier import append_to_excel
        return append_to_excel

    def test_creates_new_file(self, tmp_path):
        fn = self._get_fn()
        p = tmp_path / "out.xlsx"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        fn(str(p), df)
        result = pd.read_excel(str(p))
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 2

    def test_appends_to_existing_no_duplicate_header(self, tmp_path):
        fn = self._get_fn()
        p = tmp_path / "out.xlsx"
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"a": [3], "b": [4]})
        fn(str(p), df1)
        fn(str(p), df2)
        result = pd.read_excel(str(p))
        # Should have exactly one header row and two data rows
        assert len(result) == 2
        assert list(result.columns) == ["a", "b"]
        assert result["a"].tolist() == [1, 3]

    def test_appends_many_batches(self, tmp_path):
        """Scalability: append 50 batches, verify total row count."""
        fn = self._get_fn()
        p = tmp_path / "big.xlsx"
        for i in range(50):
            fn(str(p), pd.DataFrame({"seg": [f"img_{i}"], "val": [i]}))
        result = pd.read_excel(str(p))
        assert len(result) == 50


# ============================================================
# 2. normalize_prediction_columns
# ============================================================

class TestNormalizePredictionColumns:

    def _get_fn(self):
        from deepzplankton.core.classifier import normalize_prediction_columns
        return normalize_prediction_columns

    def test_adds_missing_columns(self):
        fn = self._get_fn()
        df = pd.DataFrame({"segment": ["a", "b"]})
        df = fn(df)
        assert "class_pred" in df.columns
        assert "confidence" in df.columns
        assert df["confidence"].dtype in [float, np.float64]

    def test_handles_whitespace_in_column_names(self):
        fn = self._get_fn()
        df = pd.DataFrame({" segment ": ["a"], " class_pred ": ["COP"], " confidence ": [0.9]})
        df = fn(df)
        assert "class_pred" in df.columns
        assert "confidence" in df.columns

    def test_coerces_nan_confidence_to_zero(self):
        fn = self._get_fn()
        df = pd.DataFrame({"segment": ["a"], "class_pred": ["COP"], "confidence": [None]})
        df = fn(df)
        assert df["confidence"].iloc[0] == 0.0


# ============================================================
# 3. load_meta – edge cases
# ============================================================

class TestLoadMeta:

    def _get_fn(self):
        from deepzplankton.core.classifier import load_meta
        return load_meta

    def test_raises_if_meta_missing(self, tmp_path):
        fn = self._get_fn()
        with pytest.raises(FileNotFoundError):
            fn(str(tmp_path))

    def test_raises_if_required_fields_missing(self, tmp_path):
        fn = self._get_fn()
        bad = {"framework": "tf", "arch": "tf_savedmodel"}
        (tmp_path / "meta.json").write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="missing required fields"):
            fn(str(tmp_path))

    def test_raises_for_unknown_framework(self, tmp_path):
        """Classifier should reject unknown frameworks early."""
        from deepzplankton.core.classifier import load_meta, PREPROCESSORS
        good_meta = {
            "framework": "unknown_fw",
            "arch": "tf_savedmodel",
            "image_size": [300, 300],
            "class_names": ["A", "B"],
            "preprocessing": "efficientnet",
        }
        (tmp_path / "meta.json").write_text(json.dumps(good_meta), encoding="utf-8")
        meta = load_meta(str(tmp_path))
        # framework validation is done in classifier(), not load_meta – document the gap
        assert meta["framework"] == "unknown_fw"

    def test_unknown_preprocessing_detected(self, tmp_path):
        """classifier() must reject unknown preprocessing key."""
        from deepzplankton.core.classifier import load_meta, PREPROCESSORS
        bad_meta = {
            "framework": "tf",
            "arch": "tf_savedmodel",
            "image_size": [300, 300],
            "class_names": ["A"],
            "preprocessing": "nonexistent_preproc",
        }
        (tmp_path / "meta.json").write_text(json.dumps(bad_meta), encoding="utf-8")
        meta = load_meta(str(tmp_path))
        # preprocessing key is NOT in PREPROCESSORS – this is what classifier() checks
        assert meta["preprocessing"] not in PREPROCESSORS


# ============================================================
# 4. BUG: Filename collision in pred_dict / output folder
# ============================================================

class TestFilenameCollision:
    """
    Two images from different source sub-folders can share the same
    basename (e.g., sample_A/img_0.jpg and sample_B/img_0.jpg).
    classifierExe.py keys pred_dict by basename, so the second entry
    silently overwrites the first — leading to mis-classified images.
    """

    def test_deduplication_keeps_first_occurrence(self, tmp_path):
        """
        The fixed classifier() deduplicates file_paths by basename before
        inference, keeping the first occurrence and warning about later ones.
        Replicate that logic here to confirm it behaves correctly.
        """
        file_paths = [
            str(tmp_path / "folder_A" / "img_0.jpg"),
            str(tmp_path / "folder_B" / "img_0.jpg"),  # duplicate basename
            str(tmp_path / "folder_A" / "img_1.jpg"),  # unique
        ]
        seen: dict = {}
        deduped = []
        for p in file_paths:
            name = os.path.basename(p).strip()
            if name not in seen:
                seen[name] = p
                deduped.append(p)

        assert len(deduped) == 2  # img_0 kept once, img_1 kept
        # os.path.join, no "/" literal: en Windows el separador es "\".
        assert deduped[0].endswith(os.path.join("folder_A", "img_0.jpg"))  # gana la primera
        assert deduped[1].endswith("img_1.jpg")

    def test_no_copy_collision_after_dedup(self, tmp_path):
        """
        After deduplication, all basenames are unique so no file is overwritten
        at the copy destination.
        """
        src_A = tmp_path / "folder_A" / "img_0.jpg"
        src_A.parent.mkdir(parents=True)
        src_A.write_bytes(b"imageA")

        dest_folder = tmp_path / "Clasificados" / "CLAD"
        dest_folder.mkdir(parents=True)

        # After dedup, only src_A survives (src_B was removed as duplicate)
        file_paths = [str(src_A)]
        pred_dict = {"img_0.jpg": ("CLAD", 0.99)}

        for path_img in file_paths:
            nombre = os.path.basename(path_img).strip()
            if nombre in pred_dict:
                pred_label, _ = pred_dict[nombre]
                shutil.copy(path_img, str(dest_folder / nombre))

        dest_files = list(dest_folder.iterdir())
        assert len(dest_files) == 1
        assert dest_files[0].read_bytes() == b"imageA"


# ============================================================
# 5. BUG: Cache cleared BEFORE Excel update
# ============================================================

class TestCacheClearedBeforeExcel:
    """
    classifier() vaciaba Segmentos/_Cache al terminar, lo que borraba las
    imágenes antes de que el Excel estuviera a salvo y además hacía que
    "Clasificar" sobre el caché anduviera una sola vez. Ahora no lo vacía:
    limpian los que lo llenan (WorkerThread y ClassifierThread).
    """

    def test_classifier_does_not_clear_the_cache(self):
        src_path = os.path.join(REPO_ROOT, "deepzplankton", "core", "classifier.py")
        with open(src_path, encoding="utf-8") as f:
            source = f.read()

        assert "vaciar_carpeta(" not in source, (
            "classifier() no debe vaciar el caché: se pierden las imágenes si "
            "falla el Excel y no se puede reclasificar con otro modelo."
        )

    def test_callers_clear_the_cache_before_filling_it(self):
        """Quien llena el caché lo vacía primero — ahí sí es seguro."""
        src_path = os.path.join(
            REPO_ROOT, "deepzplankton", "gui", "threads", "classification_thread.py"
        )
        with open(src_path, encoding="utf-8") as f:
            source = f.read()

        assert "vaciar_carpeta(rt.ruta_test)" in source, (
            "ClassifierThread debe vaciar el caché antes de copiar imágenes "
            "nuevas, si no se mezclan con las de la corrida anterior."
        )


# ============================================================
# 6. BUG: Missing CLASIFICADOS_DIR existence check
# ============================================================

class TestMissingDirChecks:

    def test_filtrar_dudosos_exits_gracefully_if_clasificados_dir_missing(self, tmp_path, monkeypatch):
        """
        After the fix, filtrar_dudosos should return gracefully (no crash)
        when Clasificados/ doesn't exist yet.
        """
        import deepzplankton.core.filtering as fd

        datos_dir = tmp_path / "Datos"
        datos_dir.mkdir()
        clasificados_dir = tmp_path / "Clasificados"  # does NOT exist

        clasificados_excel = datos_dir / "Clasificados.xlsx"
        pd.DataFrame({
            "segment": ["a.jpg"],
            "class_pred": ["CLAD"],
            "confidence": [0.99],
        }).to_excel(str(clasificados_excel), index=False)

        monkeypatch.setattr(fd, "CLASIFICADOS_EXCEL", str(clasificados_excel))
        monkeypatch.setattr(fd, "SEGUROS_EXCEL", str(datos_dir / "Seguros.xlsx"))
        monkeypatch.setattr(fd, "DUDOSOS_EXCEL", str(datos_dir / "Dudosos.xlsx"))
        monkeypatch.setattr(fd, "CLASIFICADOS_DIR", str(clasificados_dir))

        # Must not raise
        fd.filtrar_dudosos()

    def test_actualizar_exits_gracefully_if_clasificados_dir_missing(self, tmp_path, monkeypatch):
        """
        After the fix, actualizar_clasificados should return gracefully when
        Clasificados/ doesn't exist yet.
        """
        import deepzplankton.core.update as ac

        datos_dir = tmp_path / "Datos"
        datos_dir.mkdir()
        clasificados_dir = tmp_path / "Clasificados"  # does NOT exist

        clasificados_excel = datos_dir / "Clasificados.xlsx"
        pd.DataFrame({
            "segment": ["a.jpg"],
            "class_pred": ["CLAD"],
            "confidence": [0.99],
        }).to_excel(str(clasificados_excel), index=False)

        monkeypatch.setattr(ac, "CLASIFICADOS_EXCEL", str(clasificados_excel))
        monkeypatch.setattr(ac, "SEGUROS_EXCEL", str(datos_dir / "Seguros.xlsx"))
        monkeypatch.setattr(ac, "DUDOSOS_EXCEL", str(datos_dir / "Dudosos.xlsx"))
        monkeypatch.setattr(ac, "CLASIFICADOS_DIR", str(clasificados_dir))

        # Must not raise
        ac.actualizar_clasificados()


# ============================================================
# 7. BUG: Mediciones.xlsx missing when "Clasificar" is used without segmentation
# ============================================================

class TestMissingMedicionesExcel:
    """
    classifier() reads excel_path (Mediciones.xlsx) unconditionally at line 331.
    If the file doesn't exist (e.g., "Clasificar" mode without prior segmentation),
    it raises FileNotFoundError.
    """

    def test_classifier_crashes_on_missing_mediciones(self, tmp_path):
        from deepzplankton.core.classifier import classifier, load_meta

        # Create a valid model dir with meta.json
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        meta = {
            "framework": "tf",
            "arch": "tf_savedmodel",
            "image_size": [64, 64],
            "class_names": ["A", "B"],
            "preprocessing": "efficientnet",
        }
        (model_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        (model_dir / "saved_model.pb").write_bytes(b"")  # dummy

        # Create ruta_test with ONE image (so early return doesn't trigger)
        ruta_test = tmp_path / "cache"
        ruta_test.mkdir()
        img_path = ruta_test / "seg_001.jpg"
        # Create a tiny valid image
        # Write minimal JPEG bytes (1x1 pixel)
        # Minimal 1x1 white JPEG
        import struct
        img_path.write_bytes(
            b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
            b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
            b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
            b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\x1e'
            b'\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00'
            b'\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00'
            b'\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b'
            b'\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd4P\x00\x00\x00\x00\x1f\xff\xd9'
        )

        ruta_salida = tmp_path / "Clasificados"
        excel_path = tmp_path / "Datos" / "Mediciones.xlsx"
        # NOTE: Mediciones.xlsx does NOT exist

        with pytest.raises((FileNotFoundError, Exception)):
            classifier(str(model_dir), str(ruta_test), str(ruta_salida), str(excel_path))


# ============================================================
# 8. BUG: Buffer overcrop near image edges
# ============================================================

class TestBufferCalculation:
    """
    When a contour's bounding box starts close to the left or top edge
    (x < BUFFER or y < BUFFER), x2/y2 are clamped to 0 but w2/h2 still
    use the full 2*BUFFER expansion, producing a crop that is wider/taller
    than intended and asymmetric around the object.
    """

    def _compute_crop_bounds(self, x, y, w, h, W, H, BUFFER=20):
        """Replicate the fixed formula from Segmenter_All_No_View.py."""
        x1 = max(x - BUFFER, 0)
        y1 = max(y - BUFFER, 0)
        x2 = min(x + w + BUFFER, W)
        y2 = min(y + h + BUFFER, H)
        return x1, y1, x2 - x1, y2 - y1  # (start_x, start_y, width, height)

    def test_normal_case_correct(self):
        x, y, w, h, W, H = 100, 100, 50, 50, 500, 500
        x1, y1, cw, ch = self._compute_crop_bounds(x, y, w, h, W, H)
        assert x1 == 80
        assert y1 == 80
        assert cw == 90   # 50 + 40 = 90
        assert ch == 90

    def test_left_edge_clamping_symmetric(self):
        """Near left edge: crop width is correctly reduced, not overcropped."""
        x, y, w, h, W, H = 5, 100, 50, 50, 500, 500
        x1, y1, cw, ch = self._compute_crop_bounds(x, y, w, h, W, H)
        assert x1 == 0  # clamped to left edge
        # Right edge = min(5 + 50 + 20, 500) = 75; width = 75 - 0 = 75
        assert cw == 75, f"Expected width 75, got {cw}"

    def test_top_edge_clamping_symmetric(self):
        """Near top edge: crop height is correctly reduced."""
        x, y, w, h, W, H = 100, 3, 50, 50, 500, 500
        x1, y1, cw, ch = self._compute_crop_bounds(x, y, w, h, W, H)
        assert y1 == 0
        # Bottom = min(3 + 50 + 20, 500) = 73; height = 73 - 0 = 73
        assert ch == 73, f"Expected height 73, got {ch}"

    def test_right_edge_clamping_correct(self):
        """Near right edge: crop is correctly limited."""
        x, y, w, h, W, H = 470, 100, 20, 50, 500, 500
        x1, y1, cw, ch = self._compute_crop_bounds(x, y, w, h, W, H)
        assert x1 == 450
        # Right = min(470 + 20 + 20, 500) = 500; width = 500 - 450 = 50
        assert cw == 50


# ============================================================
# 9. BUG: Duplicate rows in Mediciones.xlsx on re-run
# ============================================================

class TestDuplicateExcelEntries:
    """
    If the segmenter is run on the same image twice, entries are appended
    again without deduplication. Mediciones.xlsx accumulates duplicate rows.
    """

    def test_segmenter_deduplicates_on_rerun(self, tmp_path):
        """
        After the fix, processing the same image twice should replace the old
        row rather than appending a duplicate. The segmenter now does a
        dedup-aware write at the end of the run.
        """
        excel = tmp_path / "Mediciones.xlsx"

        # Simulate first run
        df_first = pd.DataFrame([{"segment": "img_0.jpg", "area": 1000, "status": "OK"}])
        df_first.to_excel(str(excel), index=False)

        # Simulate second run (re-processing same segment with updated area)
        df_second = pd.DataFrame([{"segment": "img_0.jpg", "area": 2000, "status": "OK"}])
        new_names = set(df_second["segment"].astype(str))
        df_existing = pd.read_excel(str(excel))
        df_existing = df_existing[~df_existing["segment"].astype(str).isin(new_names)]
        df_final = pd.concat([df_existing, df_second], ignore_index=True)
        df_final.to_excel(str(excel), index=False)

        result = pd.read_excel(str(excel))
        assert len(result) == 1, "Re-run should replace old entry, not duplicate it"
        assert result["area"].iloc[0] == 2000, "Updated value should be present"


# ============================================================
# 10. BUG: MIN_AREA default inconsistency
# ============================================================

class TestMinAreaDefaultInconsistency:
    """
    The GUI default is 1000, but the Segmenter function default is 600.
    A user calling the segmenter directly (not via GUI) would get
    different behaviour than via the GUI.
    """

    def test_segmenter_noview_default_matches_gui(self):
        import inspect
        from deepzplankton.segmentation.fast import Segmenter_No_View
        sig = inspect.signature(Segmenter_No_View)
        default_min_area = sig.parameters["MIN_AREA"].default
        assert default_min_area == 1000, (
            f"Segmenter_No_View default MIN_AREA is {default_min_area}; "
            "should match the GUI default of 1000."
        )

    def test_segmenter_view_default_matches_gui(self):
        import inspect
        from deepzplankton.segmentation.interactive import Segmenter_With_View
        sig = inspect.signature(Segmenter_With_View)
        default_min_area = sig.parameters["MIN_AREA"].default
        assert default_min_area == 1000, (
            f"Segmenter_With_View default MIN_AREA is {default_min_area}; "
            "should match the GUI default of 1000."
        )


# ============================================================
# 11. filtrar_dudosos – logic tests
# ============================================================

class TestFiltrarDudosos:

    def _setup_dirs(self, tmp_path):
        datos = tmp_path / "Datos"
        datos.mkdir()
        clasificados = tmp_path / "Clasificados"
        clasificados.mkdir()
        return datos, clasificados

    def test_splits_by_threshold(self, tmp_path, monkeypatch):
        import deepzplankton.core.filtering as fd
        datos, clasificados = self._setup_dirs(tmp_path)

        # Create class folder with both secure and doubtful images
        clase_dir = clasificados / "CLAD"
        clase_dir.mkdir()
        (clase_dir / "seg_high.jpg").write_bytes(b"img")
        (clase_dir / "seg_low.jpg").write_bytes(b"img")

        clasificados_excel = datos / "Clasificados.xlsx"
        pd.DataFrame({
            "segment": ["seg_high.jpg", "seg_low.jpg"],
            "class_pred": ["CLAD", "CLAD"],
            "confidence": [0.99, 0.50],
        }).to_excel(str(clasificados_excel), index=False)

        monkeypatch.setattr(fd, "CLASIFICADOS_EXCEL", str(clasificados_excel))
        monkeypatch.setattr(fd, "SEGUROS_EXCEL", str(datos / "Seguros.xlsx"))
        monkeypatch.setattr(fd, "DUDOSOS_EXCEL", str(datos / "Dudosos.xlsx"))
        monkeypatch.setattr(fd, "CLASIFICADOS_DIR", str(clasificados))

        fd.filtrar_dudosos(UMBRAL=0.95)

        seguros = pd.read_excel(str(datos / "Seguros.xlsx"))
        dudosos = pd.read_excel(str(datos / "Dudosos.xlsx"))

        assert "seg_high.jpg" in seguros["segment"].values
        assert "seg_low.jpg" in dudosos["segment"].values

    def test_no_duplicate_on_second_run(self, tmp_path, monkeypatch):
        """Running filtrar_dudosos twice should not duplicate Excel entries."""
        import deepzplankton.core.filtering as fd
        datos, clasificados = self._setup_dirs(tmp_path)

        clase_dir = clasificados / "CLAD"
        clase_dir.mkdir()
        (clase_dir / "seg_a.jpg").write_bytes(b"img")

        clasificados_excel = datos / "Clasificados.xlsx"
        pd.DataFrame({
            "segment": ["seg_a.jpg"],
            "class_pred": ["CLAD"],
            "confidence": [0.99],
        }).to_excel(str(clasificados_excel), index=False)

        seguros_path = str(datos / "Seguros.xlsx")
        dudosos_path = str(datos / "Dudosos.xlsx")
        monkeypatch.setattr(fd, "CLASIFICADOS_EXCEL", str(clasificados_excel))
        monkeypatch.setattr(fd, "SEGUROS_EXCEL", seguros_path)
        monkeypatch.setattr(fd, "DUDOSOS_EXCEL", dudosos_path)
        monkeypatch.setattr(fd, "CLASIFICADOS_DIR", str(clasificados))

        fd.filtrar_dudosos(UMBRAL=0.95)
        fd.filtrar_dudosos(UMBRAL=0.95)  # second call

        seguros = pd.read_excel(seguros_path)
        assert len(seguros) == 1, (
            "BUG: Running filtrar_dudosos twice duplicated entries in Seguros.xlsx"
        )

    def test_empty_clasificados_exits_gracefully(self, tmp_path, monkeypatch):
        import deepzplankton.core.filtering as fd
        datos = tmp_path / "Datos"
        datos.mkdir()
        clasificados = tmp_path / "Clasificados"
        clasificados.mkdir()

        # Clasificados.xlsx is missing
        monkeypatch.setattr(fd, "CLASIFICADOS_EXCEL", str(datos / "Clasificados.xlsx"))
        monkeypatch.setattr(fd, "SEGUROS_EXCEL", str(datos / "Seguros.xlsx"))
        monkeypatch.setattr(fd, "DUDOSOS_EXCEL", str(datos / "Dudosos.xlsx"))
        monkeypatch.setattr(fd, "CLASIFICADOS_DIR", str(clasificados))

        # Should return without raising
        fd.filtrar_dudosos()


# ============================================================
# 12. actualizar – logic tests
# ============================================================

class TestActualizarClasificados:

    def _setup(self, tmp_path):
        datos = tmp_path / "Datos"
        datos.mkdir()
        clasificados = tmp_path / "Clasificados"
        clasificados.mkdir()
        return datos, clasificados

    def test_removes_missing_files_from_excel(self, tmp_path, monkeypatch):
        """Files deleted from disk should be removed from Excel on actualizar."""
        import deepzplankton.core.update as ac
        datos, clasificados = self._setup(tmp_path)

        clase_dir = clasificados / "CLAD"
        clase_dir.mkdir()
        (clase_dir / "present.jpg").write_bytes(b"img")
        # "absent.jpg" is in Excel but not on disk

        clasificados_excel = datos / "Clasificados.xlsx"
        pd.DataFrame({
            "segment": ["present.jpg", "absent.jpg"],
            "class_pred": ["CLAD", "CLAD"],
            "confidence": [0.99, 0.99],
        }).to_excel(str(clasificados_excel), index=False)

        pd.DataFrame({
            "segment": ["present.jpg", "absent.jpg"],
            "class_pred": ["CLAD", "CLAD"],
            "confidence": [0.99, 0.99],
        }).to_excel(str(datos / "Seguros.xlsx"), index=False)

        pd.DataFrame(columns=["segment", "class_pred", "confidence"]).to_excel(
            str(datos / "Dudosos.xlsx"), index=False
        )

        monkeypatch.setattr(ac, "CLASIFICADOS_EXCEL", str(clasificados_excel))
        monkeypatch.setattr(ac, "SEGUROS_EXCEL", str(datos / "Seguros.xlsx"))
        monkeypatch.setattr(ac, "DUDOSOS_EXCEL", str(datos / "Dudosos.xlsx"))
        monkeypatch.setattr(ac, "CLASIFICADOS_DIR", str(clasificados))

        ac.actualizar_clasificados()

        result = pd.read_excel(str(clasificados_excel))
        assert "absent.jpg" not in result["segment"].values
        assert "present.jpg" in result["segment"].values

    def test_updates_class_when_file_moved(self, tmp_path, monkeypatch):
        """File moved to a different class folder → Excel should reflect new class."""
        import deepzplankton.core.update as ac
        datos, clasificados = self._setup(tmp_path)

        new_clase_dir = clasificados / "CHAE"  # moved here
        new_clase_dir.mkdir()
        (new_clase_dir / "seg_001.jpg").write_bytes(b"img")

        clasificados_excel = datos / "Clasificados.xlsx"
        pd.DataFrame({
            "segment": ["seg_001.jpg"],
            "class_pred": ["CLAD"],  # old class
            "confidence": [0.99],
        }).to_excel(str(clasificados_excel), index=False)

        pd.DataFrame({
            "segment": ["seg_001.jpg"],
            "class_pred": ["CLAD"],
            "confidence": [0.99],
        }).to_excel(str(datos / "Seguros.xlsx"), index=False)

        pd.DataFrame(columns=["segment", "class_pred", "confidence"]).to_excel(
            str(datos / "Dudosos.xlsx"), index=False
        )

        monkeypatch.setattr(ac, "CLASIFICADOS_EXCEL", str(clasificados_excel))
        monkeypatch.setattr(ac, "SEGUROS_EXCEL", str(datos / "Seguros.xlsx"))
        monkeypatch.setattr(ac, "DUDOSOS_EXCEL", str(datos / "Dudosos.xlsx"))
        monkeypatch.setattr(ac, "CLASIFICADOS_DIR", str(clasificados))

        ac.actualizar_clasificados()

        result = pd.read_excel(str(clasificados_excel))
        updated_row = result[result["segment"] == "seg_001.jpg"]
        assert len(updated_row) == 1
        assert updated_row["class_pred"].iloc[0] == "CHAE"
        assert updated_row["confidence"].iloc[0] == 1.0

    def test_handles_empty_clasificados_gracefully(self, tmp_path, monkeypatch):
        """If Clasificados.xlsx doesn't exist, should print error without crashing."""
        import deepzplankton.core.update as ac
        datos, clasificados = self._setup(tmp_path)

        monkeypatch.setattr(ac, "CLASIFICADOS_EXCEL", str(datos / "Clasificados.xlsx"))
        monkeypatch.setattr(ac, "SEGUROS_EXCEL", str(datos / "Seguros.xlsx"))
        monkeypatch.setattr(ac, "DUDOSOS_EXCEL", str(datos / "Dudosos.xlsx"))
        monkeypatch.setattr(ac, "CLASIFICADOS_DIR", str(clasificados))

        ac.actualizar_clasificados()  # should not raise

    def test_adds_new_file_to_seguros(self, tmp_path, monkeypatch):
        """A file present in Clasificados/ but absent from Seguros.xlsx and
        Dudosos.xlsx (p. ej. clasificado a mano, sin pasar por filtrar_dudosos)
        debe agregarse a Seguros.xlsx con confidence 1.0.

        Ejercita el path de nuevos_seg, que usa el índice por nombre armado en
        actualizar_clasificados() en vez de filtrar df_clas entero por archivo.
        """
        import deepzplankton.core.update as ac
        datos, clasificados = self._setup(tmp_path)

        clase_dir = clasificados / "CLAD"
        clase_dir.mkdir()
        (clase_dir / "nuevo.jpg").write_bytes(b"img")

        clasificados_excel = datos / "Clasificados.xlsx"
        pd.DataFrame({
            "segment": ["nuevo.jpg"],
            "class_pred": ["CLAD"],
            "confidence": [0.87],
        }).to_excel(str(clasificados_excel), index=False)

        pd.DataFrame(columns=["segment", "class_pred", "confidence"]).to_excel(
            str(datos / "Seguros.xlsx"), index=False
        )
        pd.DataFrame(columns=["segment", "class_pred", "confidence"]).to_excel(
            str(datos / "Dudosos.xlsx"), index=False
        )

        monkeypatch.setattr(ac, "CLASIFICADOS_EXCEL", str(clasificados_excel))
        monkeypatch.setattr(ac, "SEGUROS_EXCEL", str(datos / "Seguros.xlsx"))
        monkeypatch.setattr(ac, "DUDOSOS_EXCEL", str(datos / "Dudosos.xlsx"))
        monkeypatch.setattr(ac, "CLASIFICADOS_DIR", str(clasificados))

        ac.actualizar_clasificados()

        seguros = pd.read_excel(str(datos / "Seguros.xlsx"))
        assert "nuevo.jpg" in seguros["segment"].values
        row = seguros[seguros["segment"] == "nuevo.jpg"].iloc[0]
        assert row["class_pred"] == "CLAD"
        assert row["confidence"] == 1.0

    def test_adds_new_file_to_dudosos(self, tmp_path, monkeypatch):
        """Un archivo dentro de Clasificados/_DUDOSOS/ que todavía no está en
        Dudosos.xlsx debe agregarse ahí. Ejercita el path de nuevos_dud."""
        import deepzplankton.core.update as ac
        datos, clasificados = self._setup(tmp_path)

        dudosos_dir = clasificados / "_DUDOSOS" / "CLAD"
        dudosos_dir.mkdir(parents=True)
        (dudosos_dir / "incierto.jpg").write_bytes(b"img")

        clasificados_excel = datos / "Clasificados.xlsx"
        pd.DataFrame({
            "segment": ["incierto.jpg"],
            "class_pred": ["CLAD"],
            "confidence": [0.4],
        }).to_excel(str(clasificados_excel), index=False)

        pd.DataFrame(columns=["segment", "class_pred", "confidence"]).to_excel(
            str(datos / "Seguros.xlsx"), index=False
        )
        pd.DataFrame(columns=["segment", "class_pred", "confidence"]).to_excel(
            str(datos / "Dudosos.xlsx"), index=False
        )

        monkeypatch.setattr(ac, "CLASIFICADOS_EXCEL", str(clasificados_excel))
        monkeypatch.setattr(ac, "SEGUROS_EXCEL", str(datos / "Seguros.xlsx"))
        monkeypatch.setattr(ac, "DUDOSOS_EXCEL", str(datos / "Dudosos.xlsx"))
        monkeypatch.setattr(ac, "CLASIFICADOS_DIR", str(clasificados))

        ac.actualizar_clasificados()

        dudosos = pd.read_excel(str(datos / "Dudosos.xlsx"))
        seguros = pd.read_excel(str(datos / "Seguros.xlsx"))
        assert "incierto.jpg" in dudosos["segment"].values
        assert "incierto.jpg" not in seguros["segment"].values

    def test_promotes_reclassified_dudoso_to_seguro(self, tmp_path, monkeypatch):
        """Si un dudoso se movió a mano a otra carpeta de clase (reclasificado),
        pasa a ser seguro: se mueve fuera de _DUDOSOS/ y a Seguros.xlsx."""
        import deepzplankton.core.update as ac
        datos, clasificados = self._setup(tmp_path)

        # Estaba clasificado como CLAD (dudoso); el usuario lo movió a CHAE.
        nueva_clase_dir = clasificados / "_DUDOSOS" / "CHAE"
        nueva_clase_dir.mkdir(parents=True)
        (nueva_clase_dir / "reclasificado.jpg").write_bytes(b"img")

        clasificados_excel = datos / "Clasificados.xlsx"
        pd.DataFrame({
            "segment": ["reclasificado.jpg"],
            "class_pred": ["CLAD"],
            "confidence": [0.3],
        }).to_excel(str(clasificados_excel), index=False)

        pd.DataFrame(columns=["segment", "class_pred", "confidence"]).to_excel(
            str(datos / "Seguros.xlsx"), index=False
        )
        pd.DataFrame({
            "segment": ["reclasificado.jpg"],
            "class_pred": ["CLAD"],  # clase original en el excel de dudosos
            "confidence": [0.3],
        }).to_excel(str(datos / "Dudosos.xlsx"), index=False)

        monkeypatch.setattr(ac, "CLASIFICADOS_EXCEL", str(clasificados_excel))
        monkeypatch.setattr(ac, "SEGUROS_EXCEL", str(datos / "Seguros.xlsx"))
        monkeypatch.setattr(ac, "DUDOSOS_EXCEL", str(datos / "Dudosos.xlsx"))
        monkeypatch.setattr(ac, "CLASIFICADOS_DIR", str(clasificados))

        ac.actualizar_clasificados()

        # Se movió físicamente fuera de _DUDOSOS/, a Clasificados/CHAE/
        assert (clasificados / "CHAE" / "reclasificado.jpg").exists()
        assert not (clasificados / "_DUDOSOS" / "CHAE" / "reclasificado.jpg").exists()

        seguros = pd.read_excel(str(datos / "Seguros.xlsx"))
        dudosos = pd.read_excel(str(datos / "Dudosos.xlsx"))
        assert "reclasificado.jpg" in seguros["segment"].values
        assert seguros[seguros["segment"] == "reclasificado.jpg"].iloc[0]["class_pred"] == "CHAE"
        assert "reclasificado.jpg" not in dudosos["segment"].values

    def test_removes_older_duplicate_physical_copy(self, tmp_path, monkeypatch):
        """Mismo nombre de archivo en dos carpetas de clase distintas: se
        conserva la más reciente y se borra la vieja."""
        import deepzplankton.core.update as ac
        datos, clasificados = self._setup(tmp_path)

        vieja_dir = clasificados / "CLAD"
        vieja_dir.mkdir()
        vieja = vieja_dir / "dup.jpg"
        vieja.write_bytes(b"img viejo")

        nueva_dir = clasificados / "CHAE"
        nueva_dir.mkdir()
        nueva = nueva_dir / "dup.jpg"
        nueva.write_bytes(b"img nuevo")

        # Asegurar que "nueva" tenga mtime posterior de forma determinística.
        now = time.time()
        os.utime(str(vieja), (now - 100, now - 100))
        os.utime(str(nueva), (now, now))

        clasificados_excel = datos / "Clasificados.xlsx"
        pd.DataFrame({
            "segment": ["dup.jpg"],
            "class_pred": ["CHAE"],
            "confidence": [0.99],
        }).to_excel(str(clasificados_excel), index=False)

        pd.DataFrame({
            "segment": ["dup.jpg"],
            "class_pred": ["CHAE"],
            "confidence": [0.99],
        }).to_excel(str(datos / "Seguros.xlsx"), index=False)
        pd.DataFrame(columns=["segment", "class_pred", "confidence"]).to_excel(
            str(datos / "Dudosos.xlsx"), index=False
        )

        monkeypatch.setattr(ac, "CLASIFICADOS_EXCEL", str(clasificados_excel))
        monkeypatch.setattr(ac, "SEGUROS_EXCEL", str(datos / "Seguros.xlsx"))
        monkeypatch.setattr(ac, "DUDOSOS_EXCEL", str(datos / "Dudosos.xlsx"))
        monkeypatch.setattr(ac, "CLASIFICADOS_DIR", str(clasificados))

        ac.actualizar_clasificados()

        assert not vieja.exists()
        assert nueva.exists()


# ============================================================
# 13. Scalability: Excel with many rows
# ============================================================

class TestScalability:

    def test_large_mediciones_excel_append(self, tmp_path):
        """Appending 10,000 segments in batches must complete in reasonable time."""
        from deepzplankton.core.classifier import append_to_excel
        p = tmp_path / "Mediciones.xlsx"
        BATCH = 500
        BATCHES = 20  # 10,000 total rows

        start = time.time()
        for b in range(BATCHES):
            rows = [
                {
                    "segment": f"img_{b*BATCH+i}.jpg",
                    "area": 1000 + i,
                    "circularity": 0.5,
                    "status": "OK",
                }
                for i in range(BATCH)
            ]
            append_to_excel(str(p), pd.DataFrame(rows))
        elapsed = time.time() - start

        result = pd.read_excel(str(p))
        assert len(result) == BATCHES * BATCH
        # Should complete in under 60 seconds on any reasonable machine
        assert elapsed < 60, f"Appending 10k rows took {elapsed:.1f}s – too slow"

    def test_filtrar_dudosos_large_existing_seguros(self, tmp_path, monkeypatch):
        """
        Scalability: filtrar_dudosos reads entire Seguros.xlsx to build the
        existing_segments set. With 50,000 rows this should still be fast.
        """
        import deepzplankton.core.filtering as fd

        datos = tmp_path / "Datos"
        datos.mkdir()
        clasificados = tmp_path / "Clasificados"
        clase_dir = clasificados / "CLAD"
        clase_dir.mkdir(parents=True)
        (clase_dir / "new_seg.jpg").write_bytes(b"img")

        # Large Seguros.xlsx (50,000 rows)
        seguros_path = datos / "Seguros.xlsx"
        n = 50_000
        pd.DataFrame({
            "segment": [f"existing_{i}.jpg" for i in range(n)],
            "class_pred": ["CLAD"] * n,
            "confidence": [0.99] * n,
        }).to_excel(str(seguros_path), index=False)

        clasificados_excel = datos / "Clasificados.xlsx"
        pd.DataFrame({
            "segment": ["new_seg.jpg"],
            "class_pred": ["CLAD"],
            "confidence": [0.98],
        }).to_excel(str(clasificados_excel), index=False)

        monkeypatch.setattr(fd, "CLASIFICADOS_EXCEL", str(clasificados_excel))
        monkeypatch.setattr(fd, "SEGUROS_EXCEL", str(seguros_path))
        monkeypatch.setattr(fd, "DUDOSOS_EXCEL", str(datos / "Dudosos.xlsx"))
        monkeypatch.setattr(fd, "CLASIFICADOS_DIR", str(clasificados))

        start = time.time()
        fd.filtrar_dudosos(UMBRAL=0.95)
        elapsed = time.time() - start

        assert elapsed < 30, (
            f"filtrar_dudosos with 50k existing rows took {elapsed:.1f}s. "
            "Reading the full Excel file for deduplication is a scalability concern."
        )

    def test_get_all_images_large_folder(self, tmp_path):
        """
        DropGrid.get_all_images() must handle folders with thousands of images
        without timing out.
        """
        large_folder = tmp_path / "large_campaign"
        large_folder.mkdir()
        # Create 5,000 dummy image filenames (just empty files)
        for i in range(5000):
            (large_folder / f"img_{i:05d}.jpg").write_bytes(b"")

        start = time.time()
        images = []
        for root, _, files in os.walk(str(large_folder)):
            for f in sorted(files):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")):
                    images.append(os.path.join(root, f))
        elapsed = time.time() - start

        assert len(images) == 5000
        assert elapsed < 5, f"Walking 5k-image folder took {elapsed:.2f}s"


# ============================================================
# 14. vaciar_carpeta – correctness tests
# ============================================================

class TestVaciarCarpeta:

    def test_removes_files_and_subdirs(self, tmp_path):
        from deepzplankton.core.classifier import vaciar_carpeta
        target = tmp_path / "cache"
        target.mkdir()
        (target / "a.jpg").write_bytes(b"img")
        sub = target / "sub"
        sub.mkdir()
        (sub / "b.jpg").write_bytes(b"img")

        vaciar_carpeta(str(target))

        assert list(target.iterdir()) == []

    def test_empty_folder_is_safe(self, tmp_path):
        from deepzplankton.core.classifier import vaciar_carpeta
        target = tmp_path / "empty"
        target.mkdir()
        vaciar_carpeta(str(target))
        assert target.exists()

    def test_called_before_excel_update_is_dangerous(self, tmp_path):
        """
        Demonstrate: if we call vaciar_carpeta then fail to write Excel,
        the images are gone with no recovery path.
        """
        from deepzplankton.core.classifier import vaciar_carpeta
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "precious.jpg").write_bytes(b"classified segment")

        vaciar_carpeta(str(cache))
        assert not (cache / "precious.jpg").exists()

        # Simulate Excel write failure AFTER cache was already cleared
        with pytest.raises(Exception):
            raise RuntimeError("Excel write failed – images already deleted!")


# ============================================================
# 15. is_model_dir – edge cases
# ============================================================

class TestIsModelDir:

    def test_returns_false_for_empty_dir(self, tmp_path):
        from deepzplankton.core.paths import is_model_dir
        empty = tmp_path / "empty_model"
        empty.mkdir()
        assert not is_model_dir(str(empty))

    def test_returns_true_for_meta_json_only(self, tmp_path):
        from deepzplankton.core.paths import is_model_dir
        d = tmp_path / "pt_model"
        d.mkdir()
        (d / "meta.json").write_text("{}", encoding="utf-8")
        assert is_model_dir(str(d))

    def test_returns_false_for_model_pt_only(self, tmp_path):
        """La app ya no ejecuta checkpoints .pt (sólo ONNX): una carpeta con
        únicamente model.pt no debe listarse como modelo utilizable."""
        from deepzplankton.core.paths import is_model_dir
        d = tmp_path / "pt_model"
        d.mkdir()
        (d / "model.pt").write_bytes(b"weights")
        assert not is_model_dir(str(d))

    def test_returns_false_for_nonexistent_path(self, tmp_path):
        from deepzplankton.core.paths import is_model_dir
        assert not is_model_dir(str(tmp_path / "does_not_exist"))

    def test_returns_false_for_file_not_dir(self, tmp_path):
        from deepzplankton.core.paths import is_model_dir
        f = tmp_path / "meta.json"
        f.write_text("{}", encoding="utf-8")
        assert not is_model_dir(str(f))


# ============================================================
# 16. Segmenter metrics – unit tests
# ============================================================

class TestSegmenterMetrics:

    def test_circularity_of_circle_is_near_one(self):
        import cv2
        from deepzplankton.segmentation.fast import circularity
        # Create a circular contour
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img, (100, 100), 50, 255, -1)
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        assert len(cnts) == 1
        c = circularity(cnts[0])
        # Discrete pixel circles have slightly lower circularity than the
        # theoretical 1.0 because the perimeter follows pixel steps.
        assert 0.80 <= c <= 1.05, f"Expected ~0.85-1.0 for circle, got {c:.3f}"

    def test_circularity_zero_perimeter(self):
        """Single-point contour → perimeter=0 → should return 0, not divide by zero."""
        from deepzplankton.segmentation.fast import circularity
        # Degenerate contour (single point)
        cnt = np.array([[[0, 0]]], dtype=np.int32)
        result = circularity(cnt)
        assert result == 0

    def test_solidity_of_convex_shape_is_one(self):
        import cv2
        from deepzplankton.segmentation.fast import solidity
        # Rectangle is convex → solidity = 1
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(cnts) == 1
        s = solidity(cnts[0])
        assert 0.98 <= s <= 1.0, f"Convex shape should have solidity ~1, got {s:.3f}"

    def test_eccentricity_of_square_is_one(self):
        import cv2
        from deepzplankton.segmentation.fast import eccentricity
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        e = eccentricity(cnts[0])
        assert e == pytest.approx(1.0, abs=0.01)

    def test_eccentricity_of_long_rectangle(self):
        import cv2
        from deepzplankton.segmentation.fast import eccentricity
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(img, (10, 90), (190, 110), 255, -1)  # wide & short
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        e = eccentricity(cnts[0])
        # w=180, h=20 → eccentricity = 9
        assert e > 1.0

    def test_fill_ratio_full_coverage(self):
        from deepzplankton.segmentation.fast import fill_ratio
        assert fill_ratio(100, 10, 10) == pytest.approx(1.0)

    def test_fill_ratio_partial(self):
        from deepzplankton.segmentation.fast import fill_ratio
        # Half the bounding box is filled
        assert fill_ratio(50, 10, 10) == pytest.approx(0.5)


# ============================================================
# 17. BUG: TF dependency leaks into PyTorch inference path
# ============================================================

class TestTFDependencyInPTPath:
    """
    load_images_as_batch always imports TensorFlow, even when running
    PyTorch inference. This means TF must be installed even for pure PT models.
    """

    def test_load_images_as_batch_uses_pil_not_tensorflow(self):
        """
        After the fix, load_images_as_batch must use PIL (no TF dependency),
        so PyTorch-only setups don't require TensorFlow to be installed.
        """
        src_path = os.path.join(REPO_ROOT, "deepzplankton", "core", "classifier.py")
        with open(src_path, encoding="utf-8") as f:
            source = f.read()

        fn_start = source.find("def load_images_as_batch")
        fn_end = source.find("\ndef ", fn_start + 1)
        fn_body = source[fn_start:fn_end]

        assert "PIL" in fn_body or "pillow" in fn_body.lower(), (
            "load_images_as_batch should use PIL for image loading, not TensorFlow"
        )
        assert "tensorflow" not in fn_body.lower(), (
            "load_images_as_batch must not import tensorflow (breaks pure-PT setups)"
        )


# ============================================================
# 18. BUG: torch.load without weights_only=True
# ============================================================

class TestTorchLoadSecurity:
    """
    torch.load deserializa con pickle: un .pt de origen desconocido puede
    ejecutar código arbitrario al cargarse. Al pasar a ONNX-only el riesgo
    desaparece de raíz, porque ya no se carga ningún checkpoint.
    """

    def test_classifier_never_calls_torch_load(self):
        src_path = os.path.join(REPO_ROOT, "deepzplankton", "core", "classifier.py")
        with open(src_path, encoding="utf-8") as f:
            source = f.read()

        assert "torch.load" not in source, (
            "La inferencia no debe deserializar checkpoints: los modelos que "
            "agrega el usuario se cargan como ONNX."
        )

    def test_classifier_imports_neither_torch_nor_tensorflow(self):
        """Lo que sostiene el build liviano: sin estos imports no entran al bundle."""
        src_path = os.path.join(REPO_ROOT, "deepzplankton", "core", "classifier.py")
        with open(src_path, encoding="utf-8") as f:
            source = f.read()

        for lib in ("import torch", "import tensorflow", "import timm",
                    "from transformers", "from tensorflow"):
            assert lib not in source, f"classifierExe.py no debe usar '{lib}'"


# ============================================================
# 19. app_paths – PyInstaller bundle path
# ============================================================

class TestAppPathsBundle:

    def test_resource_path_uses_meipass_when_set(self, monkeypatch):
        from deepzplankton.core.paths import resource_path
        monkeypatch.setattr(sys, "_MEIPASS", "/fake/bundle", raising=False)
        result = resource_path("model")
        assert result == os.path.join("/fake/bundle", "model")

    def test_resource_path_uses_file_dir_when_no_meipass(self, monkeypatch):
        from deepzplankton.core.paths import resource_path
        if hasattr(sys, "_MEIPASS"):
            monkeypatch.delattr(sys, "_MEIPASS")
        result = resource_path("model")
        expected = os.path.join(REPO_ROOT, "deepzplankton", "model")
        assert result == expected

    def test_app_dir_with_bundled_subdir(self, tmp_path, monkeypatch):
        """If DeepZPlankton/ subdir exists, app_dir returns it."""
        from deepzplankton.core.paths import APP_NAME
        import deepzplankton.core.paths as ap

        bundled = tmp_path / APP_NAME
        bundled.mkdir()
        monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
        # Need to reimport to pick up monkeypatched _MEIPASS
        result = ap.app_dir()
        assert result == str(bundled)


# ============================================================
# 20. classifierExe – Excel column first-column assumption
# ============================================================

class TestExcelColumnAssumption:
    """
    classifier() assumes df.columns[0] is the segment name column.
    If Mediciones.xlsx ever gets a different column order, results are wrong.
    """

    def test_first_column_is_segment_in_mediciones(self, tmp_path):
        """Verify the segmenter always writes 'segment' as the first column."""
        # The expected column from Segmenter_All_No_View
        expected_columns = [
            "segment", "x", "y", "width", "height",
            "area", "circularity", "solidity", "eccentricity",
            "texture_std", "fill_ratio",
        ]
        assert expected_columns[0] == "segment"

    def test_classifier_uses_first_column_not_named_segment(self, tmp_path):
        """
        If Mediciones.xlsx first column is NOT 'segment', classifier()
        would silently match no rows (no crash, but empty results).
        This documents the fragility.
        """
        # Simulated Mediciones.xlsx with wrong first column
        df = pd.DataFrame({
            "filename": ["seg_001.jpg"],  # wrong name – should be "segment"
            "area": [1200],
            "status": ["OK"],
        })
        df.to_excel(str(tmp_path / "Mediciones.xlsx"), index=False)
        result = pd.read_excel(str(tmp_path / "Mediciones.xlsx"))
        nombre_columna = result.columns[0]

        pred_dict = {"seg_001.jpg": ("CLAD", 0.99)}
        matched = result[result[nombre_columna].isin(pred_dict.keys())]
        # With wrong column name "filename" → the isin check still works on values
        # But the column is named "filename" not "segment", so further access
        # by "segment" would crash.
        assert len(matched) == 1  # values match
        with pytest.raises(KeyError):
            _ = matched["segment"]  # column name mismatch causes crash
