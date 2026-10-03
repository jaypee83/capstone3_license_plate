"""End-to-end ANPR: image -> plate box -> characters -> text."""
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .detect import PlateCandidate, crop_plate, find_plate
from .postprocess import Decoded, decode
from .recognize import predict_proba
from .segment import Segmentation, segment_characters


@dataclass
class ANPRResult:
    image: np.ndarray
    plate_found: bool
    detection: Optional[PlateCandidate]
    plate_crop: np.ndarray
    segmentation: Segmentation
    probs: np.ndarray
    decoded: Decoded
    timings_ms: Dict[str, float] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        """Mean top-1 probability of the characters (0-1)."""
        return float(self.probs.max(1).mean()) if len(self.probs) else 0.0


def run(image_bgr: np.ndarray, mode: str = "auto", use_format: bool = True) -> ANPRResult:
    """mode: 'auto' = find the plate in a photo (falls back to the whole image),
             'plate' = the image is already a cropped plate."""
    t = {}
    t0 = time.perf_counter()
    det = None
    crop = image_bgr
    if mode == "auto":
        det = find_plate(image_bgr)
        if det is not None and det.method != "full image":
            crop = crop_plate(image_bgr, det)
    t["detect"] = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    seg = segment_characters(crop)
    if mode == "plate" and len(seg.chars) < 4:
        # The user said "cropped plate" but nothing plate-like was found: try detection instead.
        det = find_plate(image_bgr)
        if det is not None and det.method != "full image":
            crop = crop_plate(image_bgr, det)
            seg = segment_characters(crop)
    t["segment"] = (time.perf_counter() - t1) * 1000
    t2 = time.perf_counter()
    probs = predict_proba(seg.chars)
    t["recognize"] = (time.perf_counter() - t2) * 1000
    dec = decode(probs, use_format=use_format)
    t["total"] = (time.perf_counter() - t0) * 1000
    found = det is not None and det.method != "full image"
    return ANPRResult(image_bgr, found, det if found else None, crop, seg, probs, dec, t)


def annotate(result: ANPRResult) -> np.ndarray:
    """Draw the plate box and the read text on a copy of the input image."""
    vis = result.image.copy()
    if result.detection is not None:
        x, y, w, h = result.detection.box
        th = max(2, vis.shape[1] // 300)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), th)
        label = result.decoded.pretty() or "?"
        fs = max(0.6, vis.shape[1] / 900)
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        y0 = max(tht + 8, y - 8)
        cv2.rectangle(vis, (x, y0 - tht - 8), (x + tw + 8, y0 + 4), (0, 200, 0), -1)
        cv2.putText(vis, label, (x + 4, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 2, cv2.LINE_AA)
    return vis


def char_boxes_image(result: ANPRResult) -> np.ndarray:
    vis = result.segmentation.plate.copy()
    for (x, y, w, h) in result.segmentation.boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return vis
