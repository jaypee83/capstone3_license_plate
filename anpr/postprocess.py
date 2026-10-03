"""Indian number-plate format decoding.

Format: SS NN X(XX) NNNN, e.g. MH 12 AB 1234
  SS   = state / UT code (2 letters)
  NN   = district RTO number (1-2 digits)
  XXX  = series (0-3 letters)
  NNNN = number (4 digits)
Bharat series: YY BH NNNN X(X), e.g. 22 BH 1234 AA

The CNN often confuses look-alikes (0/O/D, 1/I, 8/B, 5/S, 2/Z). Instead of taking the
single most likely character at each position, we pick the format layout whose
characters have the highest joint probability, using the CNN's softmax outputs. This
is a small form of constrained decoding.
"""
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import CLASSES

DIGIT_IDX = [i for i, c in enumerate(CLASSES) if c.isdigit()]
LETTER_IDX = [i for i, c in enumerate(CLASSES) if c.isalpha()]

STATE_CODES = {
    "AN": "Andaman & Nicobar", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh", "AS": "Assam",
    "BR": "Bihar", "CG": "Chhattisgarh", "CH": "Chandigarh", "DD": "Dadra & Nagar Haveli and Daman & Diu",
    "DL": "Delhi", "DN": "Dadra & Nagar Haveli (old code)", "GA": "Goa", "GJ": "Gujarat", "HP": "Himachal Pradesh",
    "HR": "Haryana", "JH": "Jharkhand", "JK": "Jammu & Kashmir", "KA": "Karnataka", "KL": "Kerala",
    "LA": "Ladakh", "LD": "Lakshadweep", "MH": "Maharashtra", "ML": "Meghalaya", "MN": "Manipur",
    "MP": "Madhya Pradesh", "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha", "OR": "Odisha (old code)",
    "PB": "Punjab", "PY": "Puducherry", "RJ": "Rajasthan", "SK": "Sikkim", "TG": "Telangana",
    "TN": "Tamil Nadu", "TR": "Tripura", "TS": "Telangana", "UA": "Uttarakhand (old code)",
    "UK": "Uttarakhand", "UP": "Uttar Pradesh", "WB": "West Bengal",
}

STANDARD_RE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$")
BH_RE = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")


@dataclass
class Decoded:
    raw: str                      # argmax at every position, no format knowledge
    text: str                     # best format-consistent reading
    valid: bool                   # text matches an Indian plate format
    layout: str                   # e.g. "LLDDLLDDDD"
    state: Optional[str]          # state name, if a state code was read
    corrected: List[int]          # positions changed by the decoder

    def pretty(self) -> str:
        t = self.text
        m = re.match(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{0,3})([0-9]{4})$", t)
        if m:
            return " ".join(g for g in m.groups() if g)
        m = re.match(r"^([0-9]{2})(BH)([0-9]{4})([A-Z]{1,2})$", t)
        return " ".join(m.groups()) if m else t


def _layouts(n: int):
    out = []
    for d in (1, 2):
        for l in range(0, 4):
            if 2 + d + l + 4 == n:
                out.append("LL" + "D" * d + "L" * l + "DDDD")
    for l in (1, 2):                       # Bharat series YY BH NNNN X(X)
        if 2 + 2 + 4 + l == n:
            out.append("DDLL" + "DDDD" + "L" * l)
    return out


def _best_in(row: np.ndarray, kind: str):
    idx = DIGIT_IDX if kind == "D" else LETTER_IDX
    j = idx[int(np.argmax(row[idx]))]
    return CLASSES[j], float(np.log(row[j] + 1e-9))


def decode(probs: np.ndarray, use_format: bool = True) -> Decoded:
    raw = "".join(CLASSES[i] for i in probs.argmax(1)) if len(probs) else ""
    if not use_format or len(probs) == 0:
        return Decoded(raw, raw, bool(STANDARD_RE.match(raw) or BH_RE.match(raw)), "", STATE_CODES.get(raw[:2]), [])
    best = None
    for lay in _layouts(len(probs)):
        chars, score = [], 0.0
        for row, kind in zip(probs, lay):
            c, s = _best_in(row, kind)
            chars.append(c)
            score += s
        if lay.startswith("LL"):
            # the first two letters must form a real state/UT code
            st_best, st_score = None, -1e9
            for code in STATE_CODES:
                s = np.log(probs[0][CLASSES.index(code[0])] + 1e-9) + np.log(probs[1][CLASSES.index(code[1])] + 1e-9)
                if s > st_score:
                    st_best, st_score = code, s
            score += st_score - sum(_best_in(probs[i], "L")[1] for i in range(2))
            chars[0], chars[1] = st_best[0], st_best[1]
        else:
            if "".join(chars[2:4]) != "BH":
                score += np.log(probs[2][CLASSES.index("B")] + 1e-9) + np.log(probs[3][CLASSES.index("H")] + 1e-9) \
                         - sum(_best_in(probs[i], "L")[1] for i in (2, 3))
                chars[2], chars[3] = "B", "H"
        if best is None or score > best[0]:
            best = (score, "".join(chars), lay)
    if best is None:   # length does not fit any Indian layout
        return Decoded(raw, raw, False, "", STATE_CODES.get(raw[:2]), [])
    _, text, lay = best
    corrected = [i for i, (a, b) in enumerate(zip(raw, text)) if a != b]
    valid = bool(STANDARD_RE.match(text) or BH_RE.match(text))
    state = STATE_CODES.get(text[:2]) if lay.startswith("LL") else "Bharat series (all-India)"
    return Decoded(raw, text, valid, lay, state, corrected)
