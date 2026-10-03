"""One normalisation function used everywhere: training data, synthetic data and live
inference. Keeping a single function is what stops 'train/serve skew', where the model
sees differently-prepared characters at test time than during training."""
import cv2
import numpy as np

from .config import CHAR_INNER, CHAR_SIZE


def normalize_char(mask: np.ndarray) -> np.ndarray:
    """mask: 2-D uint8, character pixels > 0 (white on black).
    Returns a 28x28 uint8 image: tight crop -> stretch to 24x24 -> 2 px black border.
    (The original Kaggle dataset stores characters stretched to fill the frame,
    so we stretch too, instead of preserving aspect ratio.)"""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    ys, xs = np.nonzero(mask)
    out = np.zeros((CHAR_SIZE, CHAR_SIZE), np.uint8)
    if len(xs) == 0:
        return out
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    crop = cv2.resize(crop, (CHAR_INNER, CHAR_INNER), interpolation=cv2.INTER_AREA)
    pad = (CHAR_SIZE - CHAR_INNER) // 2
    out[pad:pad + CHAR_INNER, pad:pad + CHAR_INNER] = crop
    _, out = cv2.threshold(out, 100, 255, cv2.THRESH_BINARY)
    return out


def to_tensor_batch(chars) -> np.ndarray:
    """List of 28x28 uint8 -> float32 array (N, 1, 28, 28) scaled to [0, 1]."""
    if len(chars) == 0:
        return np.zeros((0, 1, CHAR_SIZE, CHAR_SIZE), np.float32)
    return (np.stack(chars).astype(np.float32) / 255.0)[:, None, :, :]
