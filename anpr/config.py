"""Shared constants and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SAMPLES = ROOT / "samples"

CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")   # 36 classes, same order as the dataset folders
DIGITS = set("0123456789")
LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

CHAR_SIZE = 28            # CNN input is 28x28, 1 channel, white character on black
CHAR_INNER = 24           # the glyph is stretched to 24x24 and padded with 2 px of black
PLATE_HEIGHT = 100        # plates are resized to this height before segmentation

CNN_ONNX = MODELS / "char_cnn.onnx"
HAAR_XML = MODELS / "indian_license_plate.xml"
METRICS_JSON = MODELS / "metrics.json"
