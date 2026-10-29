"""Pipeline de detección de contornos compartido por los dos segmentadores."""
import cv2
import numpy as np


def detectar_contornos(gray):
    """gray (imagen en escala de grises) -> lista de contornos externos."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def expandir_bbox(x, y, w, h, buffer, W, H):
    """Expande el bounding box `buffer` px por lado, recortado a los bordes
    de la imagen. Se calculan primero los extremos para no sobre-recortar
    cuando el inicio queda pisado contra el borde."""
    x1 = max(x - buffer, 0)
    y1 = max(y - buffer, 0)
    x2 = min(x + w + buffer, W)
    y2 = min(y + h + buffer, H)
    return x1, y1, x2, y2
