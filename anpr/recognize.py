"""Character recognition with the trained CNN, exported to ONNX and run by OpenCV's DNN
module. This keeps the deployed app light: no TensorFlow or PyTorch needed at runtime."""
from functools import lru_cache
from typing import List

import cv2
import numpy as np

from .config import CLASSES, CNN_ONNX
from .preprocess import to_tensor_batch


@lru_cache(maxsize=1)
def _net(path: str = str(CNN_ONNX)):
    return cv2.dnn.readNetFromONNX(path)


def predict_proba(chars: List[np.ndarray]) -> np.ndarray:
    """chars: list of 28x28 uint8 -> (N, 36) class probabilities (the model ends in softmax)."""
    if not chars:
        return np.zeros((0, len(CLASSES)), np.float32)
    net = _net()
    out = []
    for x in to_tensor_batch(chars):          # one at a time: robust across OpenCV versions
        net.setInput(x[None])
        out.append(net.forward()[0])
    return np.array(out, np.float32)


def top_k(probs_row: np.ndarray, k: int = 3):
    idx = np.argsort(-probs_row)[:k]
    return [(CLASSES[i], float(probs_row[i])) for i in idx]
