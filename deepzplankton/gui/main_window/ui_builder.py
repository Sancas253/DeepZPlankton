import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QFont, QIntValidator
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QMenu, QPushButton,
    QProgressBar, QSplitter, QTextEdit, QVBoxLayout,
)

from ..streams import EmittingStream
from ..widgets.drop_grid import DropGrid


class _UiBuilderMixin:
    """Arma el layout de MainWindow. Separado de window.py para que la
    ventana principal no sea, además de todo lo demás, 200 líneas de
    construcción de widgets."""

    def _build_ui(self):

        self.setWindowTitle("DeepZPlankton")
        self.resize(1100, 900)

        layout = QVBoxLayout(self)

        title = QLabel("DeepZPlankton")
        title.setFont(QFont("Segoe UI", 28, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2D6CDF; margin-top: 10px;")

        subtitle = QLabel("Clasificación automática de zooplancton")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #78909C; font-size: 12pt; margin-bottom: 6px;")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        # ===== SPLITTER VERTICAL =====

        splitter_vertical = QSplitter(Qt.Vertical)

        self.dropgrid = DropGrid(self)

        # ===== SPLITTER INFERIOR =====

        splitter_inferior = QSplitter(Qt.Horizontal)

        # LOG
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        # PANEL CONTROLES
        panel = self._build_panel_controles()

        splitter_inferior.addWidget(self.log)
        splitter_inferior.addWidget(panel)
        splitter_inferior.setSizes([700, 250])

        splitter_vertical.addWidget(self.dropgrid)
        splitter_vertical.addWidget(splitter_inferior)
        splitter_vertical.setSizes([600, 300])

        layout.addWidget(splitter_vertical)

        # ===== PROGRESS BAR =====

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #E0E0E0;
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background-color: #4A90D9;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        layout.addLayout(self._build_button_bar())

        self._stdout_stream = EmittingStream()
        self._stdout_stream.text_written.connect(self.log_print)
        sys.stdout = self._stdout_stream

        self._stderr_stream = EmittingStream()
        self._stderr_stream.text_written.connect(self.log_print)
        sys.stderr = self._stderr_stream

        self.cargar_modelos()

    def _build_panel_controles(self):
        from PyQt5.QtWidgets import QLineEdit, QVBoxLayout, QWidget

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)

        # Área mínima
        panel_layout.addWidget(QLabel("Área mínima (px²):"))

        self.input_area = QLineEdit("1000")
        self.input_area.setValidator(QIntValidator(1, 999999))
        panel_layout.addWidget(self.input_area)

        # Umbral
        panel_layout.addWidget(QLabel("Umbral filtrar dudosos:"))

        umbral_row = QHBoxLayout()
        umbral_row.setSpacing(4)
        self.input_umbral = QLineEdit("0.95")
        self.input_umbral.setValidator(QDoubleValidator(0.0, 1.0, 3))
        self.btn_refilter = QPushButton("↺")
        self.btn_refilter.setFixedWidth(30)
        self.btn_refilter.setToolTip("Reaplicar umbral a toda la base clasificada")
        self.btn_refilter.setStyleSheet(
            "QPushButton { padding: 2px 2px; font-size: 13pt; min-height: 0px; }"
        )
        self.btn_refilter.clicked.connect(self.iniciar_refilter)
        umbral_row.addWidget(self.input_umbral)
        umbral_row.addWidget(self.btn_refilter)
        panel_layout.addLayout(umbral_row)

        # Modelo
        panel_layout.addWidget(QLabel("Modelo:"))

        self.combo_modelos = QComboBox()
        panel_layout.addWidget(self.combo_modelos)

        self.btn_add_model = QPushButton("Agregar modelo")
        add_menu = QMenu(self.btn_add_model)
        add_menu.addAction("Carpeta...",     self._agregar_desde_carpeta)
        add_menu.addAction("Archivo ZIP...", self._agregar_desde_zip)
        self.btn_add_model.setMenu(add_menu)
        panel_layout.addWidget(self.btn_add_model)

        panel_layout.addStretch()
        return panel

    def _build_button_bar(self):
        button_bar = QHBoxLayout()

        self.btn_cargar = QPushButton("Cargar")
        cargar_menu = QMenu(self.btn_cargar)
        cargar_menu.addAction("Imágenes", self.cargar_imagenes)
        cargar_menu.addAction("Carpeta", self.cargar_carpeta)
        self.btn_cargar.setMenu(cargar_menu)

        self.btn_rapido = QPushButton("Segmentar rápido")
        self.btn_rapido.clicked.connect(lambda: self.iniciar_proceso("rapido"))

        self.btn_detalle = QPushButton("Segmentar detalle")
        self.btn_detalle.clicked.connect(lambda: self.iniciar_proceso("detalle"))

        self.btn_clasificar = QPushButton("Clasificar")
        self.btn_clasificar.clicked.connect(self.iniciar_clasificacion)

        self.chk_mejor_fold = QCheckBox("Solo mejor modelo")
        self.chk_mejor_fold.setToolTip(
            "Usar solo el mejor fold en lugar del ensemble completo.\n"
            "Aprox. 5× más rápido con mínima pérdida de precisión."
        )
        self.chk_mejor_fold.setChecked(False)

        self.btn_actualizar = QPushButton("Actualizar")
        self.btn_actualizar.setToolTip(
            "Confirma los resultados de EnRevision/ hacia Historico/ y, a\n"
            "continuación, sincroniza los Excel con lo que haya en Historico/\n"
            "(reclasificaciones hechas a mano). Cada parte se saltea si no\n"
            "tiene nada que hacer."
        )
        self.btn_actualizar.clicked.connect(self.iniciar_actualizar)

        self.btn_borrar = QPushButton("Vaciar cargadas")
        self.btn_borrar.setToolTip(
            "Quita las imágenes cargadas en pantalla. No borra nada del disco\n"
            "(ni EnRevision/ ni Historico/)."
        )
        self.btn_borrar.clicked.connect(self.dropgrid.clear_all)

        # Sólo cancela segmentación/clasificación (WorkerThread y
        # ClassifierThread): son las corridas largas y desatendidas sobre
        # campañas grandes. Asentar/Refiltrar son operaciones de archivo más
        # cortas y no tienen puntos de corte seguros instrumentados.
        self.btn_cancelar = QPushButton("Cancelar")
        self.btn_cancelar.setEnabled(False)
        self.btn_cancelar.clicked.connect(self.cancelar_proceso)
        self._active_worker = None

        button_bar.addWidget(self.btn_cargar)
        button_bar.addWidget(self.btn_rapido)
        button_bar.addWidget(self.btn_detalle)
        button_bar.addWidget(self.btn_clasificar)
        button_bar.addWidget(self.chk_mejor_fold)
        button_bar.addWidget(self.btn_actualizar)
        button_bar.addWidget(self.btn_borrar)
        button_bar.addWidget(self.btn_cancelar)

        return button_bar
