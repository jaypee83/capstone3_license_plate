import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anpr.config import CLASSES, SAMPLES  # noqa: E402
from anpr.pipeline import run  # noqa: E402
from anpr.postprocess import decode  # noqa: E402
from anpr.preprocess import normalize_char  # noqa: E402
from anpr.recognize import predict_proba  # noqa: E402
from anpr.segment import segment_characters  # noqa: E402
from anpr.synth import TEST_FONT_CANDIDATES, TRAIN_FONT_CANDIDATES, find_fonts, random_plate_text, render_plate  # noqa: E402


def test_real_plate_is_read_exactly():
    r = run(cv2.imread(str(SAMPLES / "real_plate_DL8CAF5030.png")), mode="plate")
    assert r.decoded.text == "DL8CAF5030" and r.decoded.valid and r.decoded.state == "Delhi"


@pytest.mark.parametrize("name", ["MH12AB1234", "KA05MN2024", "DL3CAQ7788"])
def test_plate_found_and_read_in_car_scene(name):
    r = run(cv2.imread(str(SAMPLES / f"synthetic_car_{name}.jpg")), mode="auto")
    assert r.plate_found and r.decoded.text == name


@pytest.mark.parametrize("name", ["TN09BX4321", "22BH4512AA"])
def test_cropped_plate_in_auto_mode_is_not_cut(name):
    assert run(cv2.imread(str(SAMPLES / f"synthetic_plate_{name}.jpg")), mode="auto").decoded.text == name


def test_onnx_outputs_probabilities():
    p = predict_proba([np.zeros((28, 28), np.uint8)])
    assert p.shape == (1, 36) and abs(float(p.sum()) - 1) < 1e-4


def test_normalize_char_shape_and_empty_input():
    assert normalize_char(np.zeros((40, 20), np.uint8)).shape == (28, 28)
    m = np.zeros((40, 20), np.uint8)
    m[5:35, 8:12] = 255
    out = normalize_char(m)
    assert out.shape == (28, 28) and out.max() == 255 and out[:2].max() == 0   # 2 px border stays black


def onehot(text, conf=0.9):
    p = np.full((len(text), 36), (1 - conf) / 35, np.float32)
    for i, c in enumerate(text):
        p[i, CLASSES.index(c)] = conf
    return p


def test_format_decoding_fixes_lookalikes():
    p = onehot("MH12AB1234")
    p[0] = onehot("N")[0] * 0.5 + onehot("M")[0] * 0.4     # CNN slightly prefers 'N', but "NH" is not a state
    p[4] = onehot("8")[0] * 0.55 + onehot("B")[0] * 0.45   # 8/B confusion where a series LETTER must be
    d = decode(p)
    assert d.raw == "NH128B1234"            # what plain argmax reads
    assert d.text == "MH12BB1234"           # format-aware reading
    assert d.valid and d.state == "Maharashtra" and d.corrected == [0, 4]


def test_decoding_rejects_impossible_length():
    assert not decode(onehot("AB12")).valid


def test_segmentation_on_unseen_font_plates():
    fonts = find_fonts(TEST_FONT_CANDIDATES) or find_fonts(TRAIN_FONT_CANDIDATES)
    rng = np.random.default_rng(123)
    ok = 0
    for i in range(30):
        t = random_plate_text(rng)
        ok += len(segment_characters(render_plate(t, fonts[i % len(fonts)], rng)).chars) == len(t)
    assert ok >= 27
