import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QGridLayout, QLabel, QScrollArea, QWidget

from .cards import FolderWidget, ThumbnailWidget

VALID = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


class DropGrid(QScrollArea):

    def __init__(self, parent):

        super().__init__(parent)

        self.setWidgetResizable(True)
        self.setAcceptDrops(True)

        self.container = QWidget()
        self.grid = QGridLayout(self.container)
        self.grid.setSpacing(15)

        self.setWidget(self.container)

        self.cards: list = []   # list of _Card widgets
        self._selected_paths: set = set()
        self.row = 0
        self.col = 0
        self.max_cols = 4

        self.setFocusPolicy(Qt.StrongFocus)
        self._create_empty_label()

    def _create_empty_label(self):
        self.empty_label = QLabel("Arrastrá imágenes o carpetas acá\n\n📂")
        self.empty_label.setFont(QFont("Segoe UI", 16))
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("color: #90A4AE; border: none; background: transparent;")
        self.grid.addWidget(self.empty_label, 0, 0, Qt.AlignCenter)

    def _hide_empty_label(self):
        if self.empty_label:
            self.empty_label.hide()
            self.empty_label = None

    def dragEnterEvent(self, event):

        if event.mimeData().hasUrls():
            event.accept()

    def dropEvent(self, event):

        self._hide_empty_label()

        for url in event.mimeData().urls():

            path = url.toLocalFile()

            if os.path.isdir(path):
                self.add_folder(path)
            elif path.lower().endswith(VALID):
                self.add_thumbnail(path)

    def _add_widget(self, widget, path):

        self._hide_empty_label()

        widget._drop_grid = self
        widget.setFixedSize(200, 240)
        self.grid.addWidget(widget, self.row, self.col)
        self.cards.append(widget)

        self.col += 1
        if self.col >= self.max_cols:
            self.col = 0
            self.row += 1

    def add_thumbnail(self, img_path):
        self._add_widget(ThumbnailWidget(img_path), img_path)

    def add_folder(self, folder_path):
        count = sum(
            1 for r, _, fs in os.walk(folder_path)
            for f in fs if f.lower().endswith(VALID)
        )
        self._add_widget(FolderWidget(folder_path, count), folder_path)

    def get_all_images(self):
        images = []
        for card in self.cards:
            p = card.path
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for f in sorted(files):
                        if f.lower().endswith(VALID):
                            images.append(os.path.join(root, f))
            else:
                images.append(p)
        return images

    def _on_card_clicked(self, card, event):
        self.setFocus()
        if event.modifiers() & Qt.ControlModifier:
            card.set_selected(not card._selected)
            if card._selected:
                self._selected_paths.add(card.path)
            else:
                self._selected_paths.discard(card.path)
        else:
            for c in self.cards:
                c.set_selected(False)
            self._selected_paths.clear()
            card.set_selected(True)
            self._selected_paths.add(card.path)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self._selected_paths:
            self.cards = [c for c in self.cards if c.path not in self._selected_paths]
            self._selected_paths.clear()
            self._rebuild_grid()
        else:
            super().keyPressEvent(event)

    def _rebuild_grid(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item and item.widget():
                item.widget().setParent(None)

        self.row = 0
        self.col = 0

        if not self.cards:
            self._create_empty_label()
            return

        for card in self.cards:
            self.grid.addWidget(card, self.row, self.col)
            self.col += 1
            if self.col >= self.max_cols:
                self.col = 0
                self.row += 1

    def clear_all(self):

        while self.grid.count():
            item = self.grid.takeAt(0)
            if item and item.widget():
                item.widget().setParent(None)

        self.cards = []
        self._selected_paths.clear()
        self.row = 0
        self.col = 0

        self._create_empty_label()
