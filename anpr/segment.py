"""Character segmentation: plate image -> list of 28x28 character images (left to right).

Steps: resize -> grayscale -> contrast boost (CLAHE) -> Otsu threshold so characters are
white -> remove blobs touching the top/bottom border -> keep connected components whose
size and shape look like characters -> drop outliers in height/position (e.g. the IND
emblem or screws) -> sort left to right -> normalise each to 28x28.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

from .config import PLATE_HEIGHT
from .preprocess import normalize_char


@dataclass
class Segmentation:
    plate: np.ndarray                       # resized plate (BGR)
    binary: np.ndarray                      # binary image, characters white
    boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)   # x, y, w, h per character
    chars: List[np.ndarray] = field(default_factory=list)                  # 28x28 per character


def binarize(plate_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY) if plate_bgr.ndim == 3 else plate_bgr
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # If most of the plate came out white, the plate has light text on a dark background: invert.
    if binary.mean() > 127:
        binary = 255 - binary
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return binary


def segment_characters(plate_bgr: np.ndarray, min_chars: int = 1) -> Segmentation:
    h0, w0 = plate_bgr.shape[:2]
    scale = PLATE_HEIGHT / max(h0, 1)
    plate = cv2.resize(plate_bgr, (max(1, int(w0 * scale)), PLATE_HEIGHT), interpolation=cv2.INTER_CUBIC)
    binary = binarize(plate)
    H, W = binary.shape

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cand = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches_tb = y <= 1 or y + h >= H - 1
        if touches_tb and h > 0.9 * H:          # plate border / frame
            continue
        if not (0.30 * H <= h <= 0.95 * H):      # too short or too tall for a character
            continue
        if not (0.08 <= w / h <= 1.25):          # characters are taller than wide
            continue
        if area < 0.12 * w * h:                  # thin frame lines, not solid strokes
            continue
        if w > 0.35 * W:
            continue
        cand.append((x, y, w, h, i))

    if cand:
        # Remove outliers: characters on a plate share a similar height and baseline.
        hs = np.array([c[3] for c in cand], np.float32)
        cy = np.array([c[1] + c[3] / 2 for c in cand], np.float32)
        mh, mcy = np.median(hs), np.median(cy)
        cand = [c for c, hh, yy in zip(cand, hs, cy) if 0.72 * mh <= hh <= 1.3 * mh and abs(yy - mcy) <= 0.22 * H]

    cand.sort(key=lambda c: c[0])
    seg = Segmentation(plate=plate, binary=binary)
    for x, y, w, h, i in cand:
        mask = np.where(labels[y:y + h, x:x + w] == i, 255, 0).astype(np.uint8)
        seg.boxes.append((int(x), int(y), int(w), int(h)))
        seg.chars.append(normalize_char(mask))
    return seg
