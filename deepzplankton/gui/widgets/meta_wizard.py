"""Arma el meta.json de un modelo ONNX leyendo lo que se pueda del grafo."""
import os

from PyQt5.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

from ...core import onnx_probe
from ...core.preprocessing import PREPROCESSORS
from .meta_wizard_actions import _MetaWizardActionsMixin


class MetaWizardDialog(_MetaWizardActionsMixin, QDialog):
    """Sólo ONNX: _ensure_meta() descarta antes cualquier carpeta sin .onnx,
    así que acá self._onnx nunca es None."""

    # Preprocesado sugerido según cómo fue exportado el ONNX. Un grafo NHWC sale
    # de Keras/tf2onnx y lleva la normalización adentro (espera [0,255] crudo);
    # uno NCHW sale de torch.onnx y espera mean/std de ImageNet aplicado afuera.
    _LAYOUT_PREPROC = {"NHWC": "efficientnet", "NCHW": "imagenet_meanstd"}

    def __init__(self, model_dir, parent=None):
        super().__init__(parent)
        self.model_dir = model_dir
        self._onnx = onnx_probe.describe(model_dir) or {}
        self.setWindowTitle("Configurar metadatos del modelo")
        self.setMinimumWidth(460)
        self._build_ui()

    def _framework_label(self):
        n = self._onnx.get("n_folds") or 0
        return f"ONNX — ensemble de {n} folds" if n > 1 else "ONNX"

    def _build_ui(self):
        layout = QVBoxLayout(self)

        name = os.path.basename(self.model_dir)
        info = QLabel(f"<b>Modelo:</b> {name}<br><b>Framework:</b> {self._framework_label()}")
        info.setWordWrap(True)
        layout.addWidget(info)

        detectado = []
        if self._onnx.get("image_size"):
            h, w = self._onnx["image_size"]
            detectado.append(f"entrada {h}×{w} ({self._onnx['layout']})")
        if self._onnx.get("n_classes") is not None:
            detectado.append(f"{self._onnx['n_classes']} clases de salida")
        if detectado:
            hint = QLabel("🔎 Leído del grafo: " + ", ".join(detectado) + ".")
            hint.setWordWrap(True)
            hint.setStyleSheet("color:#2D6CDF; font-size:9pt;")
            layout.addWidget(hint)

        form = QFormLayout()

        # En ONNX el grafo ya trae la arquitectura adentro: este campo queda
        # sólo como referencia para el usuario, la inferencia no lo mira.
        self._arch_edit = QLineEdit(name)
        form.addRow("Arquitectura (informativa):", self._arch_edit)

        size_widget = QWidget()
        size_layout = QHBoxLayout(size_widget)
        size_layout.setContentsMargins(0, 0, 0, 0)
        self._spin_h = QSpinBox()
        self._spin_h.setRange(32, 1024)
        self._spin_w = QSpinBox()
        self._spin_w.setRange(32, 1024)
        detected_size = self._onnx.get("image_size")
        if detected_size:
            # El grafo declara un tamaño fijo: cualquier otro valor rompe en
            # tiempo de inferencia, así que no se deja editar.
            self._spin_h.setValue(detected_size[0])
            self._spin_w.setValue(detected_size[1])
            self._spin_h.setEnabled(False)
            self._spin_w.setEnabled(False)
        else:
            self._spin_h.setValue(224)
            self._spin_w.setValue(224)
        size_layout.addWidget(self._spin_h)
        size_layout.addWidget(QLabel("×"))
        size_layout.addWidget(self._spin_w)
        form.addRow("Tamaño (H × W):", size_widget)

        # La lista sale del registro real del clasificador, así no se puede
        # ofrecer acá un nombre que la inferencia después no sepa resolver.
        self._preproc_combo = QComboBox()
        self._preproc_combo.addItems(list(PREPROCESSORS))
        self._preproc_combo.setCurrentText(
            self._LAYOUT_PREPROC.get(self._onnx.get("layout"), "imagenet_meanstd")
        )
        form.addRow("Preprocesamiento:", self._preproc_combo)

        self._classes_edit = QTextEdit()
        self._classes_edit.setPlaceholderText("Un nombre por línea\nEj:\nCOP_CALAN\nMEDU\nZOEA")
        self._classes_edit.setFixedHeight(150)
        form.addRow("Clases (una por línea):", self._classes_edit)

        self._classes_count = QLabel("")
        self._classes_count.setStyleSheet("font-size:9pt;")
        self._classes_edit.textChanged.connect(self._on_classes_changed)

        btn_import = QPushButton("Cargar clases desde archivo...")
        btn_import.setStyleSheet("QPushButton{padding:4px 10px; font-size:9pt;}")
        btn_import.clicked.connect(self._importar_clases)

        classes_row = QHBoxLayout()
        classes_row.setContentsMargins(0, 0, 0, 0)
        classes_row.addWidget(self._classes_count)
        classes_row.addStretch()
        classes_row.addWidget(btn_import)
        form.addRow("", classes_row)

        self._notes_edit = QLineEdit()
        self._notes_edit.setPlaceholderText("Descripción opcional")
        form.addRow("Notas:", self._notes_edit)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Guardar meta.json")
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._on_classes_changed()
