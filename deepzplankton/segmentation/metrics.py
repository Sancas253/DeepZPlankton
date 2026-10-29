"""Métricas de forma de un contorno."""
import cv2
import numpy as np

MIN_CIRCULARITY = 0.02


def circularity(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return 0
    return (4 * np.pi * area) / (peri * peri)


def solidity(cnt):
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0
    return area / hull_area


def eccentricity(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    if min(w, h) == 0:
        return 0
    return max(w, h) / min(w, h)


def fill_ratio(area, w, h):
    return area / (w * h)
