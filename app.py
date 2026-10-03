"""Automatic Number Plate Recognition (ANPR) for Indian plates - Streamlit app.

    streamlit run app.py
"""
import json

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from anpr.config import CLASSES, METRICS_JSON, SAMPLES
from anpr.pipeline import annotate, char_boxes_image, run
from anpr.recognize import top_k

st.set_page_config(page_title="ANPR · Indian Number Plates", page_icon="🚗", layout="wide")


def rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr.ndim == 3 else img_bgr


def read_upload(file) -> np.ndarray:
    data = np.frombuffer(file.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    mode_label = st.radio("What does the image show?",
                          ["A vehicle / scene (find the plate)", "Only the number plate (already cropped)"])
    mode = "auto" if mode_label.startswith("A vehicle") else "plate"
    use_format = st.checkbox("Use Indian plate format rules", value=True,
                             help="Picks the most likely reading that fits SS NN XX NNNN and a real state code, "
                                  "using the CNN's probabilities (fixes 0/O, 1/I, 8/B, 5/S confusions).")
    show_steps = st.checkbox("Show every processing step", value=True)
    st.divider()
    st.caption("Pipeline: Haar cascade + contour plate detection → OpenCV segmentation → "
               "CNN character classifier (ONNX) → Indian-format decoding")

st.title("🚗 Automatic Number Plate Recognition")
st.write("Upload a photo of a vehicle or a number plate. The app finds the plate, splits it into characters, "
         "reads each character with a convolutional neural network, and checks the result against the Indian "
         "number-plate format.")

tab_read, tab_model, tab_about = st.tabs(["🔍 Read a plate", "📊 Model performance", "ℹ️ How it works"])

# ---------------------------------------------------------------- main tab
with tab_read:
    src = st.radio("Image source", ["Sample images", "Upload a photo", "Use the camera"], horizontal=True)
    img = None
    if src == "Sample images":
        samples = sorted(p for p in SAMPLES.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        names = [p.name for p in samples]
        pick = st.selectbox("Choose a sample", names)
        img = cv2.imread(str(SAMPLES / pick))
        if pick.startswith("real_") or pick.startswith("plate_"):
            st.caption("Tip: this sample is a cropped plate. Either mode works, because the app falls back to "
                       "the whole image when no plate is found.")
    elif src == "Upload a photo":
        f = st.file_uploader("Image (JPG / PNG)", type=["jpg", "jpeg", "png"])
        if f:
            img = read_upload(f)
    else:
        cam = st.camera_input("Point the camera at a number plate")
        if cam:
            img = read_upload(cam)

    if img is None:
        st.info("Choose a sample, upload a photo, or take one with the camera.")
        st.stop()

    # very large photos slow everything down; 1280 px wide is plenty for plate reading
    if img.shape[1] > 1280:
        s = 1280 / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    res = run(img, mode=mode, use_format=use_format)
    d = res.decoded

    c1, c2 = st.columns([3, 2])
    with c1:
        st.image(rgb(annotate(res)), caption="Input" + (" with detected plate" if res.plate_found else ""),
                 use_container_width=True)
    with c2:
        if not res.segmentation.chars:
            st.error("No characters found. Try a closer, sharper photo of the plate, or switch to "
                     "'Only the number plate' mode with a cropped image.")
        else:
            st.markdown("### Plate number")
            st.markdown(f"<div style='font-size:2.4rem;font-weight:700;letter-spacing:.08em;"
                        f"border:3px solid #222;border-radius:8px;padding:.3rem .8rem;display:inline-block;"
                        f"background:#fff;color:#111'>{d.pretty()}</div>", unsafe_allow_html=True)
            st.write("")
            if d.valid:
                st.success("✅ Valid Indian plate format" + (f" · {d.state}" if d.state else ""))
            else:
                st.warning("⚠️ Does not match the Indian plate format: check the segmentation steps below.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Characters", len(res.segmentation.chars))
            m2.metric("Avg. confidence", f"{res.confidence:.0%}")
            m3.metric("Time", f"{res.timings_ms['total']:.0f} ms")
            if use_format and d.corrected:
                st.caption(f"Raw CNN reading: `{d.raw}`. Format rules corrected position(s) "
                           f"{', '.join(str(i + 1) for i in d.corrected)}.")
            st.caption("Plate found by: " + (res.detection.method.upper() if res.detection else
                                              "none, so the whole image was read as the plate"))

    if show_steps and res.segmentation.chars:
        st.divider()
        st.markdown("#### Step by step")
        s1, s2, s3 = st.columns(3)
        s1.image(rgb(res.plate_crop), caption="1 · Plate crop", use_container_width=True)
        s2.image(res.segmentation.binary, caption="2 · Binarised (Otsu + CLAHE)", use_container_width=True)
        s3.image(rgb(char_boxes_image(res)), caption="3 · Characters found", use_container_width=True)
        st.markdown("**4 · Each character as the CNN sees it (28×28), with its top-3 predictions**")
        cols = st.columns(len(res.segmentation.chars))
        for i, (col, ch) in enumerate(zip(cols, res.segmentation.chars)):
            col.image(cv2.resize(ch, (84, 84), interpolation=cv2.INTER_NEAREST), use_container_width=True)
            t3 = top_k(res.probs[i], 3)
            col.markdown(f"**{d.text[i] if i < len(d.text) else t3[0][0]}**")
            col.caption("<br>".join(f"{c} {p:.0%}" for c, p in t3), unsafe_allow_html=True)
        st.caption(f"Timings: detection {res.timings_ms['detect']:.0f} ms · segmentation "
                   f"{res.timings_ms['segment']:.0f} ms · recognition {res.timings_ms['recognize']:.0f} ms")

# ---------------------------------------------------------------- model tab
with tab_model:
    if not METRICS_JSON.exists():
        st.info("Run `python training/train_cnn.py` to create the model metrics.")
    else:
        m = json.loads(METRICS_JSON.read_text())
        rows = []
        labels = {
            "original_val_acc": "Character accuracy · original validation set (serif font)",
            "heldout_font_char_acc": "Character accuracy · unseen plate-style fonts",
            "e2e_segmentation_ok": "Plates segmented into the right number of characters",
            "e2e_plate_exact_raw": "Whole plate read exactly · no format rules",
            "e2e_plate_exact_with_format_decoding": "Whole plate read exactly · with format rules",
        }
        for key, label in labels.items():
            rows.append({"Metric": label, **{name: f"{r[key]:.1%}" for name, r in m["models"].items()}})
        rows.append({"Metric": "Real sample plate DL8CAF5030 read as",
                     **{name: r["real_sample_plate"]["read"] for name, r in m["models"].items()}})
        rows.append({"Metric": "Parameters", **{name: f"{r['params']:,}" for name, r in m["models"].items()}})
        st.markdown("#### Baseline (reference Kaggle CNN) vs improved model")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"End-to-end tests use {m['models']['improved']['e2e_plates']} synthetic plates rendered with fonts "
                   f"never used in training ({', '.join(m['test_fonts'])}).")
        for img_name, cap in (("training_curves.png", "Training curves"), ("confusion_matrix.png",
                              "Confusion matrix of the improved model on unseen fonts")):
            p = METRICS_JSON.parents[1] / "docs" / img_name
            if p.exists():
                st.image(str(p), caption=cap)
        if m.get("top_confusions_heldout"):
            st.markdown("**Most common mistakes (unseen fonts):** " + ", ".join(
                f"{t}→{p} ({n})" for n, t, p in m["top_confusions_heldout"][:6]))

# ---------------------------------------------------------------- about tab
with tab_about:
    st.markdown("""
**1. Plate detection.** A Haar cascade trained on Indian plates (from the reference project) proposes plate
regions; a contour-based detector (bright rectangle, aspect ratio 1.8–7) is the fallback. Each candidate is
kept only if characters can be segmented inside it.

**2. Segmentation.** The plate is resized to 100 px high, contrast-enhanced (CLAHE), binarised with Otsu's
method and split into connected components. Components with character-like size and shape are kept;
outliers in height or position (screws, the IND emblem) are removed.

**3. Recognition.** Each character is normalised to 28×28 and classified by a CNN (2 conv blocks with
batch-norm, ~0.9 M parameters) into 36 classes (0–9, A–Z). The model was trained in PyTorch and exported to
ONNX, so this app only needs OpenCV.

**4. Indian-format decoding.** The CNN's probabilities are combined with the plate format
(state code, district number, series, 4-digit number) to pick the most likely valid reading.

**Data.** Character dataset and Haar cascade from the
[AI-based Indian license plate detection](https://github.com/SarthakV7/AI-based-indian-license-plate-detection)
project (GPL-3.0), extended with synthetic characters rendered in sans-serif fonts.
""")
