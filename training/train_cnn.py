"""Train the character-recognition CNN and export it to ONNX.

Trains TWO models so the improvement can be measured fairly:
  * baseline : the architecture from the reference Kaggle notebook (one 24x24 conv layer),
               trained only on the original character dataset
  * improved : a deeper CNN (4 conv layers + batch-norm), trained on the original dataset
               PLUS synthetic characters rendered in plate-style sans-serif fonts

Both are evaluated on
  1. the original validation set (216 images)
  2. characters rendered in fonts held out from training
  3. end-to-end reading of synthetic plates (held-out fonts): segmentation + CNN + decoding
  4. the real sample plate from the reference project (DL8CAF5030)

Usage (from the repository root):
    python training/train_cnn.py                 # ~3-6 min on CPU, ~1 min on a Colab GPU
    python training/train_cnn.py --epochs 25 --synthetic-per-class 600
"""
import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anpr.config import CLASSES, CNN_ONNX, METRICS_JSON  # noqa: E402
from anpr.postprocess import decode  # noqa: E402
from anpr.preprocess import normalize_char, to_tensor_batch  # noqa: E402
from anpr.segment import segment_characters  # noqa: E402
from anpr.synth import (TEST_FONT_CANDIDATES, TRAIN_FONT_CANDIDATES, find_fonts, random_plate_text,  # noqa: E402
                        render_char, render_plate)

SEED = 42


# ------------------------------------------------------------------ models
class BaselineCNN(nn.Module):
    """Same layers as the reference notebook: Conv(32, 24x24, same) -> MaxPool -> Dropout(0.4)
    -> Dense(128) -> Dense(36)."""
    def __init__(self, n=36):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 24, padding="same")
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, n)
        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        x = self.drop(F.max_pool2d(F.relu(self.conv(x)), 2))
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


class CharCNN(nn.Module):
    """Improved model: two conv blocks (conv-BN-ReLU x2 + max-pool), then a small classifier.
    About 0.9 M parameters, 28x28x1 input, 36 classes."""
    def __init__(self, n=36):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))
        self.features = nn.Sequential(block(1, 32), block(32, 64), nn.Dropout(0.25))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 256), nn.ReLU(inplace=True),
                                  nn.Dropout(0.4), nn.Linear(256, n))

    def forward(self, x):
        return self.head(self.features(x))


class WithSoftmax(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.softmax(self.m(x), dim=1)


# ------------------------------------------------------------------ data
def ensure_dataset(data_dir: Path) -> Path:
    if not (data_dir / "train").exists():
        z = ROOT / "training" / "data.zip"
        print(f"Unzipping {z} ...")
        with zipfile.ZipFile(z) as f:
            for m in f.namelist():
                if not m.startswith("__MACOSX"):
                    f.extract(m, data_dir.parent)
    return data_dir


def load_split(split_dir: Path):
    X, y = [], []
    for ci, c in enumerate(CLASSES):
        d = split_dir / f"class_{c}"
        for fn in sorted(os.listdir(d)):
            img = cv2.imread(str(d / fn), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            X.append(normalize_char((img > 127).astype(np.uint8) * 255))
            y.append(ci)
    return np.array(X), np.array(y)


def augment_original(X, y, copies, rng):
    """Small rotations / thickness changes of the original images (then re-normalised)."""
    out_x, out_y = [X], [y]
    for _ in range(copies):
        aug = []
        for img in X:
            a = cv2.copyMakeBorder(img, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=0)
            M = cv2.getRotationMatrix2D((20, 20), rng.uniform(-8, 8), rng.uniform(0.9, 1.1))
            M[0, 1] += rng.uniform(-0.12, 0.12)
            a = cv2.warpAffine(a, M, (40, 40))
            r = rng.random()
            if r < 0.25:
                a = cv2.erode(a, np.ones((2, 2), np.uint8))
            elif r < 0.5:
                a = cv2.dilate(a, np.ones((2, 2), np.uint8))
            aug.append(normalize_char(a))
        out_x.append(np.array(aug))
        out_y.append(y)
    return np.concatenate(out_x), np.concatenate(out_y)


def synth_chars(fonts, per_class, rng, augment=True):
    X, y = [], []
    for ci, c in enumerate(CLASSES):
        for k in range(per_class):
            X.append(render_char(c, fonts[k % len(fonts)], rng, augment))
            y.append(ci)
    return np.array(X), np.array(y)


# ------------------------------------------------------------------ train / eval
def train(model, X, y, Xv, yv, epochs, device, name):
    torch.manual_seed(SEED)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * ((len(X) + 127) // 128))
    Xt, yt = torch.tensor(to_tensor_batch(list(X))), torch.tensor(y)
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = correct = loss_sum = 0
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            out = model(xb)
            loss = F.cross_entropy(out, yb, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item()
            tot += len(idx)
        va = accuracy(model, Xv, yv, device)
        hist.append({"epoch": ep, "loss": loss_sum / tot, "train_acc": correct / tot, "val_acc": va})
        print(f"[{name}] epoch {ep:2d}  loss {loss_sum / tot:.4f}  train {correct / tot:.4f}  val {va:.4f}")
    return hist


@torch.no_grad()
def probs_of(model, X, device):
    model.eval()
    out = []
    xt = torch.tensor(to_tensor_batch(list(X)))
    for i in range(0, len(xt), 512):
        out.append(torch.softmax(model(xt[i:i + 512].to(device)), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 36))


def accuracy(model, X, y, device):
    return float((probs_of(model, X, device).argmax(1) == y).mean())


def plate_eval(model, plates, device):
    """End-to-end: segmentation + CNN + Indian-format decoding on whole plate images."""
    exact_raw = exact_fmt = seg_ok = 0
    char_tot = char_ok = 0
    for img, text in plates:
        seg = segment_characters(img)
        if len(seg.chars) != len(text):
            continue
        seg_ok += 1
        p = probs_of(model, seg.chars, device)
        d = decode(p)
        exact_raw += d.raw == text
        exact_fmt += d.text == text
        char_ok += sum(a == b for a, b in zip(d.text, text))
        char_tot += len(text)
    n = len(plates)
    return {"plates": n, "segmentation_ok": seg_ok / n, "plate_exact_raw": exact_raw / n,
            "plate_exact_with_format_decoding": exact_fmt / n,
            "char_accuracy_when_segmented": char_ok / max(char_tot, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--synthetic-per-class", type=int, default=400)
    ap.add_argument("--plates", type=int, default=300, help="synthetic test plates")
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    t0 = time.time()

    data = ensure_dataset(ROOT / "training" / "data")
    Xtr0, ytr0 = load_split(data / "train")
    Xva0, yva0 = load_split(data / "val")
    print(f"original dataset: train {len(Xtr0)}, val {len(Xva0)}")

    train_fonts = find_fonts(TRAIN_FONT_CANDIDATES)
    test_fonts = find_fonts(TEST_FONT_CANDIDATES) or train_fonts[-1:]
    if not train_fonts:
        raise SystemExit("No training fonts found (install fonts-dejavu or matplotlib).")
    test_fonts = [f for f in test_fonts if f not in train_fonts] or test_fonts
    print("train fonts:", [os.path.basename(f) for f in train_fonts])
    print("held-out test fonts:", [os.path.basename(f) for f in test_fonts])

    Xo, yo = augment_original(Xtr0, ytr0, copies=9, rng=rng)
    Xs, ys = synth_chars(train_fonts, args.synthetic_per_class, rng)
    Xsv, ysv = synth_chars(train_fonts, 40, rng)
    Xheld, yheld = synth_chars(test_fonts, 40, rng)            # held-out fonts
    plates = []
    for i in range(args.plates):
        t = random_plate_text(rng)
        plates.append((render_plate(t, test_fonts[i % len(test_fonts)], rng), t))
    real = cv2.imread(str(ROOT / "samples" / "real_plate_DL8CAF5030.png"))

    results = {}
    histories = {}
    models = {}
    specs = [("improved", CharCNN(), np.concatenate([Xo, Xs]), np.concatenate([yo, ys]),
              np.concatenate([Xva0, Xsv]), np.concatenate([yva0, ysv]))]
    if not args.skip_baseline:
        specs.insert(0, ("baseline", BaselineCNN(), Xo, yo, Xva0, yva0))
    for name, model, X, y, Xv, yv in specs:
        print(f"\n=== training {name} on {len(X)} images ===")
        histories[name] = train(model, X, y, Xv, yv, args.epochs, device, name)
        models[name] = model
        seg = segment_characters(real)
        real_read = decode(probs_of(model, seg.chars, device)).text if seg.chars else ""
        results[name] = {
            "train_images": int(len(X)),
            "params": int(sum(p.numel() for p in model.parameters())),
            "original_val_acc": accuracy(model, Xva0, yva0, device),
            "heldout_font_char_acc": accuracy(model, Xheld, yheld, device),
            **{f"e2e_{k}": v for k, v in plate_eval(model, plates, device).items()},
            "real_sample_plate": {"truth": "DL8CAF5030", "read": real_read, "correct": real_read == "DL8CAF5030"},
        }
        print(json.dumps(results[name], indent=1))

    # ---- export the improved model to ONNX (softmax included) and verify with OpenCV DNN
    best = models["improved"].cpu().eval()
    CNN_ONNX.parent.mkdir(exist_ok=True)
    torch.onnx.export(WithSoftmax(best), torch.zeros(1, 1, 28, 28), str(CNN_ONNX), input_names=["input"],
                      output_names=["probs"], opset_version=13, dynamo=False)
    net = cv2.dnn.readNetFromONNX(str(CNN_ONNX))
    xb = to_tensor_batch(list(Xheld[:20]))
    diff = 0.0
    for i in range(len(xb)):
        net.setInput(xb[i:i + 1])
        diff = max(diff, float(np.abs(net.forward()[0] - probs_of(best, Xheld[i:i + 1], "cpu")[0]).max()))
    print(f"ONNX export verified with OpenCV DNN, max |diff| = {diff:.2e}")
    torch.save(best.state_dict(), CNN_ONNX.with_suffix(".pt"))

    # ---- confusion matrix (improved model, held-out fonts) and training curves
    pred = probs_of(best, Xheld, "cpu").argmax(1)
    cm = np.zeros((36, 36), int)
    for t, p in zip(yheld, pred):
        cm[t, p] += 1
    confusions = sorted(((int(cm[i, j]), CLASSES[i], CLASSES[j]) for i in range(36) for j in range(36) if i != j and cm[i, j]),
                        reverse=True)[:10]
    metrics = {"models": results, "history": histories, "top_confusions_heldout": confusions,
               "classes": CLASSES, "seed": SEED, "epochs": args.epochs,
               "train_fonts": [os.path.basename(f) for f in train_fonts],
               "test_fonts": [os.path.basename(f) for f in test_fonts],
               "onnx_max_abs_diff": diff, "train_seconds": round(time.time() - t0, 1)}
    METRICS_JSON.write_text(json.dumps(metrics, indent=1))
    np.save(ROOT / "models" / "confusion_heldout.npy", cm)
    try:
        save_plots(histories, cm)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    print(f"\nSaved {CNN_ONNX} and {METRICS_JSON}  ({metrics['train_seconds']} s)")


def save_plots(histories, cm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, h in histories.items():
        e = [r["epoch"] for r in h]
        ax[0].plot(e, [r["loss"] for r in h], label=name)
        ax[1].plot(e, [r["val_acc"] for r in h], label=name)
    ax[0].set_title("Training loss"); ax[1].set_title("Validation accuracy")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend()
    fig.tight_layout(); fig.savefig(docs / "training_curves.png", dpi=130); plt.close(fig)
    fig, a = plt.subplots(figsize=(8, 7))
    a.imshow(cm, cmap="Blues")
    a.set_xticks(range(36)); a.set_xticklabels(CLASSES, fontsize=7)
    a.set_yticks(range(36)); a.set_yticklabels(CLASSES, fontsize=7)
    a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title("Confusion matrix (held-out fonts)")
    fig.tight_layout(); fig.savefig(docs / "confusion_matrix.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
