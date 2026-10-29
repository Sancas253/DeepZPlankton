"""Hoja de estilos (QSS) de la app."""

LIGHT_THEME = """
QWidget {
    background-color: #F0F2F5;
    color: #1A1D23;
    font-family: 'Segoe UI';
    font-size: 11pt;
}

QPushButton {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #4A90D9, stop:1 #2D6CDF);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 22px;
    font-weight: bold;
    font-size: 10pt;
    min-height: 20px;
}

QPushButton:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #5BA0EF, stop:1 #3C7DFF);
}

QPushButton:pressed {
    background-color: #1E4FAA;
}

QPushButton:disabled {
    background-color: #B0BEC5;
    color: #ECEFF1;
}

QLineEdit {
    background-color: white;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: #4A90D9;
}

QLineEdit:focus {
    border: 2px solid #4A90D9;
}

QTextEdit {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    padding: 8px;
    font-family: 'Consolas', 'Courier New', 'Segoe UI Emoji', monospace;
    font-size: 10pt;
}

QComboBox {
    background-color: white;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 6px 8px;
    min-height: 20px;
}

QComboBox:hover {
    border: 1px solid #4A90D9;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QScrollArea {
    border: 2px dashed #B0BEC5;
    border-radius: 12px;
    background-color: #FAFBFC;
}

QLabel {
    background-color: transparent;
    border: none;
}

QSplitter::handle {
    background-color: #CBD5E1;
    height: 3px;
    margin: 4px 0;
    border-radius: 1px;
}
"""
