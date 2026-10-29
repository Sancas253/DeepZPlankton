import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


class _Card(QWidget):
    """Selectable base for thumbnail and folder cards."""

    _STYLE_NORMAL   = "QWidget{background-color:white;border:1px solid #E0E0E0;border-radius:10px;padding:6px;}"
    _STYLE_SELECTED = "QWidget{background-color:#EBF5FF;border:2px solid #4A90D9;border-radius:10px;padding:6px;}"

    def __init__(self, path):
        super().__init__()
        self.path = path
        self._selected = False
        self._drop_grid = None

    def set_selected(self, val):
        self._selected = val
        self.setStyleSheet(self._STYLE_SELECTED if val else self._STYLE_NORMAL)

    def mousePressEvent(self, event):
        if self._drop_grid is not None:
            self._drop_grid._on_card_clicked(self, event)
        super().mousePressEvent(event)


class ThumbnailWidget(_Card):

    def __init__(self, img_path):
        from PyQt5.QtGui import QPixmap

        super().__init__(img_path)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        pix = QPixmap(img_path).scaled(
            180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        img_label = QLabel()
        img_label.setPixmap(pix)
        img_label.setAlignment(Qt.AlignCenter)

        base = os.path.basename(img_path)

        text_label = QLabel(base)
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setWordWrap(True)

        layout.addWidget(img_label)
        layout.addWidget(text_label)

        self.set_selected(False)


class FolderWidget(_Card):

    def __init__(self, folder_path, count):

        super().__init__(folder_path)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        icon_label = QLabel("📁")
        icon_label.setFont(QFont("Segoe UI", 40))
        icon_label.setAlignment(Qt.AlignCenter)

        name_label = QLabel(os.path.basename(folder_path))
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setWordWrap(True)
        name_label.setFont(QFont("Segoe UI", 10, QFont.Bold))

        count_label = QLabel(f"{count} imágenes")
        count_label.setAlignment(Qt.AlignCenter)
        count_label.setStyleSheet("color: #78909C; font-size: 9pt;")

        layout.addWidget(icon_label)
        layout.addWidget(name_label)
        layout.addWidget(count_label)

        self.set_selected(False)
