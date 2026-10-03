"""Plate localisation.

1. Haar cascade trained on Indian plates (the method used in the reference Kaggle project).
2. Fallback: a classical contour method that looks for a bright rectangle with a plate-like
   aspect ratio.
Every candidate is validated by trying to segment characters inside it; the candidate with
the most plausible characters wins. If nothing is found, the caller can treat the whole
image as the plate (useful when the user uploads an already-cropped plate).
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import HAAR_XML
from .segment import segment_characters


@dataclass
class PlateCandidate:
    box: Tuple[int, int, int, int]   # x, y, w, h in the original image
    method: str                      # "haar" or "contour"
    n_chars: int


@lru_cache(maxsize=1)
def _cascade() -> Optional[cv2.CascadeClassifier]:
    c = cv2.CascadeClassifier(str(HAAR_XML))
    return None if c.empty() else c


def _crop(img: np.ndarray, box, pad: float = 0.04) -> np.ndarray:
    x, y, w, h = box
    px, py = int(w * pad), int(h * pad)
    H, W = img.shape[:2]
    return img[max(0, y - py):min(H, y + h + py), max(0, x - px):min(W, x + w + px)]


def haar_candidates(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    c = _cascade()
    if c is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = c.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 12))
    return [tuple(int(v) for v in b) for b in boxes]


def contour_candidates(img: np.ndarray, max_candidates: int = 12) -> List[Tuple[int, int, int, int]]:
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 50, 50)
    edges = cv2.Canny(gray, 50, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:60]:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 0.004 * W * H or w * h > 0.5 * W * H:
            continue
        if not (1.8 <= w / max(h, 1) <= 7.0):
            continue
        rect_fill = cv2.contourArea(c) / (w * h)
        if rect_fill < 0.5:
            continue
        out.append((x, y, w, h))
        if len(out) >= max_candidates:
            break
    return out


def _plate_score(n: int) -> float:
    """Indian plates have 7-11 characters; prefer candidates in that range."""
    return n if 7 <= n <= 11 else n - 10


def find_plate(img: np.ndarray, min_chars: int = 4) -> Optional[PlateCandidate]:
    """Best plate region in the image. The WHOLE image is also a candidate, so an image that is
    already a cropped plate is not cut down to a partial plate by a detector box."""
    H, W = img.shape[:2]
    best: Optional[PlateCandidate] = None
    n_full = len(segment_characters(img).chars)
    if n_full >= min_chars:
        best = PlateCandidate(box=(0, 0, W, H), method="full image", n_chars=n_full)
    for method, boxes in (("haar", haar_candidates(img)), ("contour", contour_candidates(img))):
        for b in boxes:
            n = len(segment_characters(_crop(img, b)).chars)
            if n < min_chars:
                continue
            if best is None or _plate_score(n) > _plate_score(best.n_chars):
                best = PlateCandidate(box=b, method=method, n_chars=n)
        if best is not None and best.method != "full image" and 7 <= best.n_chars <= 11:
            break   # good plate from the detector, skip the slower fallback
    return best


def crop_plate(img: np.ndarray, cand: PlateCandidate) -> np.ndarray:
    return _crop(img, cand.box)
