"""Validación y guardado del meta.json armado por MetaWizardDialog."""
import json
import os

from PyQt5.QtWidgets import QFileDialog, QMessageBox


class _MetaWizardActionsMixin:

    def _parse_classes(self):
        raw = self._classes_edit.toPlainText().strip()
        return [c.strip() for c in raw.replace(",", "\n").splitlines() if c.strip()]

    def _on_classes_changed(self):
        n = len(self._parse_classes())
        esperado = self._onnx.get("n_classes") if self._onnx else None
        if esperado is None:
            self._classes_count.setText(f"{n} clases")
            self._classes_count.setStyleSheet("font-size:9pt; color:#78909C;")
        elif n == esperado:
            self._classes_count.setText(f"✔ {n} de {esperado} clases")
            self._classes_count.setStyleSheet("font-size:9pt; color:#2E7D32;")
        else:
            self._classes_count.setText(f"✖ {n} clases — el modelo espera {esperado}")
            self._classes_count.setStyleSheet("font-size:9pt; color:#C62828;")

    def _importar_clases(self):
        """Importa class_names de un .txt (uno por línea) o de otro meta.json."""
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Cargar nombres de clase", "",
            "Clases (*.txt *.json);;Todos los archivos (*)"
        )
        if not ruta:
            return
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                if ruta.lower().endswith(".json"):
                    data = json.load(f)
                    nombres = data.get("class_names") if isinstance(data, dict) else data
                    if not isinstance(nombres, list):
                        raise ValueError("el JSON no contiene una lista 'class_names'")
                else:
                    nombres = [ln.strip() for ln in f if ln.strip()]
        except Exception as e:
            QMessageBox.warning(self, "No se pudo leer", f"Error leyendo {os.path.basename(ruta)}:\n{e}")
            return
        self._classes_edit.setPlainText("\n".join(str(n) for n in nombres))

    def _save_and_accept(self):
        class_names = self._parse_classes()
        if not class_names:
            QMessageBox.warning(self, "Faltan clases", "Debes ingresar al menos un nombre de clase.")
            return

        preprocessing = self._preproc_combo.currentText()

        # Estas dos validaciones son el motivo de leer el grafo: un nº de clases
        # distinto produce etiquetas corridas sin ningún error, y un preprocesado
        # con el layout equivocado rompe o, peor, clasifica mal en silencio.
        esperado = self._onnx.get("n_classes")
        if esperado is not None and len(class_names) != esperado:
            QMessageBox.warning(
                self, "Cantidad de clases incorrecta",
                f"El modelo produce {esperado} salidas pero ingresaste "
                f"{len(class_names)} nombres de clase.\n\n"
                "Si se guarda así, las predicciones van a quedar mal "
                "etiquetadas sin dar error."
            )
            return

        from ...core.preprocessing import PREPROC_LAYOUT
        layout = self._onnx.get("layout")
        esperado_layout = PREPROC_LAYOUT.get(preprocessing)
        if layout and esperado_layout and layout != esperado_layout:
            resp = QMessageBox.question(
                self, "Preprocesamiento incompatible",
                f"El modelo recibe imágenes en formato {layout} pero "
                f"'{preprocessing}' entrega {esperado_layout}.\n\n"
                f"Lo esperable para este modelo es "
                f"'{self._LAYOUT_PREPROC.get(layout)}'.\n\n¿Guardar igual?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if resp == QMessageBox.No:
                return

        meta = {
            "framework":    "onnx",
            "arch":         self._arch_edit.text().strip() or "onnx",
            "image_size":   [self._spin_h.value(), self._spin_w.value()],
            "class_names":  class_names,
            "preprocessing": preprocessing,
        }
        notes = self._notes_edit.text().strip()
        if notes:
            meta["notes"] = notes

        meta_path = os.path.join(self.model_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        self.accept()
