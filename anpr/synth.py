"""Synthetic data: single characters (for training) and whole plates (for end-to-end testing).

Why: the public character dataset (864 training images) uses one SERIF font, but real
Indian plates use plain block (sans-serif) letters. Adding characters rendered in several
sans-serif fonts, with realistic distortions, closes that domain gap. Test plates are
rendered with fonts the model never saw during training, so the test is fair.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .preprocess import normalize_char

# Fonts used for TRAINING. Any that exist on the machine are used (Colab has DejaVu via
# matplotlib and usually Liberation; Linux distros have more).
TRAIN_FONT_CANDIDATES = [
    "DejaVuSans-Bold.ttf", "DejaVuSansCondensed-Bold.ttf", "DejaVuSansMono-Bold.ttf", "DejaVuSans.ttf",
    "LiberationSans-Bold.ttf", "LiberationMono-Bold.ttf", "LiberationSansNarrow-Bold.ttf",
    "FreeSansBold.ttf", "FreeMonoBold.ttf", "Carlito-Bold.ttf", "lmsans10-bold.otf", "Poppins-Bold.ttf",
]
# Fonts held out for TESTING only (Helvetica-like, close to real plates).
TEST_FONT_CANDIDATES = ["texgyreheroscn-bold.otf", "texgyreheros-bold.otf", "Arial Bold.ttf", "arialbd.ttf"]

_SEARCH_DIRS = ["/usr/share/fonts", "/usr/share/texmf/fonts", "/usr/local/share/fonts",
                os.path.expanduser("~/.fonts"), "C:/Windows/Fonts", "/Library/Fonts"]


def _matplotlib_font_dir() -> Optional[str]:
    try:
        import matplotlib
        return os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    except Exception:
        return None


def find_fonts(names: List[str]) -> List[str]:
    dirs = _SEARCH_DIRS + [d for d in [_matplotlib_font_dir()] if d]
    found = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for path in glob.glob(os.path.join(d, "**", "*"), recursive=True):
            base = os.path.basename(path)
            if base in names and base not in found:
                found[base] = path
    return [found[n] for n in names if n in found]


# ------------------------------------------------------------------ characters
def render_char(ch: str, font_path: str, rng: np.random.Generator, augment: bool = True) -> np.ndarray:
    """Render one character white-on-black, distort it like a real segmented character,
    and return it normalised to 28x28 exactly as live inference does."""
    size = 96
    font = ImageFont.truetype(font_path, int(size * rng.uniform(0.7, 0.85)) if augment else int(size * 0.8))
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    stroke = int(rng.integers(0, 3)) if augment else 0          # simulate bolder print
    draw.text((size / 2, size / 2), ch, fill=255, font=font, anchor="mm", stroke_width=stroke, stroke_fill=255)
    a = np.array(img)
    if augment:
        # small rotation + shear (camera tilt) and horizontal squeeze (condensed plate fonts)
        ang = rng.uniform(-7, 7)
        sh = rng.uniform(-0.15, 0.15)
        sx = rng.uniform(0.75, 1.1)
        M = cv2.getRotationMatrix2D((size / 2, size / 2), ang, 1.0)
        M[0, 1] += sh
        M[0, 0] *= sx
        a = cv2.warpAffine(a, M, (size, size))
        # low resolution capture: shrink then enlarge, blur, noise, then re-binarise
        s = int(rng.integers(14, 40))
        a = cv2.resize(cv2.resize(a, (s, s), interpolation=cv2.INTER_AREA), (size, size))
        if rng.random() < 0.5:
            a = cv2.GaussianBlur(a, (5, 5), rng.uniform(0.5, 1.8))
        a = np.clip(a.astype(np.float32) + rng.normal(0, 18, a.shape), 0, 255).astype(np.uint8)
        thr = rng.uniform(90, 170)
        a = (a > thr).astype(np.uint8) * 255
        k = np.ones((3, 3), np.uint8)
        r = rng.random()
        if r < 0.2:
            a = cv2.erode(a, k)
        elif r < 0.4:
            a = cv2.dilate(a, k)
        # keep only the largest blob plus anything close to it (drops noise specks)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(a)
        if n > 1:
            keep = np.zeros_like(a)
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            for i in range(1, n):
                if i == big or stats[i, cv2.CC_STAT_AREA] > 0.08 * stats[big, cv2.CC_STAT_AREA]:
                    keep[lab == i] = 255
            a = keep
    return normalize_char(a)


# ------------------------------------------------------------------ whole plates
STATE_CODES_SAMPLE = ["MH", "DL", "KA", "TN", "KL", "GJ", "UP", "RJ", "WB", "AP", "TS", "HR", "PB", "MP"]


def random_plate_text(rng: np.random.Generator) -> str:
    L = "ABCDEFGHJKLMNPRSTUVWXYZ"          # series letters (I and O are not issued)
    st = rng.choice(STATE_CODES_SAMPLE)
    dist = f"{rng.integers(1, 100):02d}"
    series = "".join(rng.choice(list(L), size=int(rng.integers(1, 3))))
    num = f"{rng.integers(1, 10000):04d}"
    return f"{st}{dist}{series}{num}"


def render_plate(text: str, font_path: str, rng: np.random.Generator, ind_strip: bool = True) -> np.ndarray:
    """Render an Indian-style plate (white background, black border, black text) with
    realistic camera effects. Returns a BGR image of just the plate region."""
    groups = [text[:2], text[2:4]]
    rest = text[4:]
    k = len(rest) - 4
    groups += [rest[:k], rest[k:]] if k > 0 else [rest]
    shown = " ".join(g for g in groups if g)
    H = 120
    font = ImageFont.truetype(font_path, 84)
    tw = font.getbbox(shown)[2]
    left = 70 if ind_strip else 30
    W = int(tw + left + 40)
    bg = int(rng.integers(215, 250))
    img = Image.new("RGB", (W, H), (bg, bg, bg))
    d = ImageDraw.Draw(img)
    d.rectangle([4, 4, W - 5, H - 5], outline=(20, 20, 20), width=5)
    if ind_strip:   # the blue "IND" strip with a hologram on new HSRP plates
        d.rectangle([10, 12, 48, H - 13], fill=(160, 60, 20))
        try:
            d.text((29, H - 30), "IND", fill=(240, 240, 240), font=ImageFont.truetype(font_path, 13), anchor="mm")
        except Exception:
            pass
    d.text((left, H / 2), shown, fill=(15, 15, 15), font=font, anchor="lm")
    a = np.array(img)[:, :, ::-1].copy()
    # camera effects: slight perspective, blur, brightness, noise, JPEG artefacts
    h, w = a.shape[:2]
    j = 0.04
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-j, j, (4, 2)).astype(np.float32) * np.float32([w, h])
    a = cv2.warpPerspective(a, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)
    a = cv2.GaussianBlur(a, (3, 3), rng.uniform(0.3, 1.2))
    a = np.clip(a.astype(np.float32) * rng.uniform(0.7, 1.15) + rng.normal(0, 6, a.shape), 0, 255).astype(np.uint8)
    scale = rng.uniform(0.45, 0.9)
    a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 90))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def place_in_scene(plate: np.ndarray, rng: np.random.Generator, size=(480, 640)) -> np.ndarray:
    """Paste a plate on a simple synthetic 'car front' scene (for testing plate detection)."""
    H, W = size
    scene = np.zeros((H, W, 3), np.uint8)
    top = np.array(rng.integers(90, 200, 3), np.float32)
    for y in range(H):   # sky/background gradient
        scene[y] = np.clip(top * (1 - y / H) + 60 * (y / H), 0, 255)
    body = tuple(int(c) for c in rng.integers(30, 200, 3))
    cv2.rectangle(scene, (60, int(H * 0.35)), (W - 60, H - 40), body, -1)                 # car body
    cv2.rectangle(scene, (110, int(H * 0.38)), (W - 110, int(H * 0.55)), (40, 40, 40), -1)  # grille
    for cx in (130, W - 130):
        cv2.circle(scene, (cx, int(H * 0.47)), 28, (230, 230, 210), -1)                  # headlights
    ph, pw = plate.shape[:2]
    s = min(1.0, (W * 0.42) / pw)
    p = cv2.resize(plate, None, fx=s, fy=s)
    ph, pw = p.shape[:2]
    x0 = (W - pw) // 2 + int(rng.integers(-40, 40))
    y0 = int(H * 0.62) + int(rng.integers(-15, 15))
    scene[y0:y0 + ph, x0:x0 + pw] = p
    scene = np.clip(scene.astype(np.float32) + rng.normal(0, 5, scene.shape), 0, 255).astype(np.uint8)
    return scene
