"""
Xenarch Mk20 - Production Web Edition + headless dataset runner
==============================================================
Unsupervised planetary-surface technosignature detection.

New in Mk20 (vs Mk19):
  A. LOW-INFORMATION CHIP REJECTION — a chip that is basically a flat grey or
     blown-out white field carries no geology and the VAE just encodes it to
     the middle of the latent space ("one data point"). Such chips are now
     detected up front (dynamic range, Shannon entropy, gradient + edge energy)
     and dropped before training and scoring. See chip_information_ok().
  B. LATENT-COLLAPSE FILTER — after encoding, any chip whose latent mean sits
     right on top of the population centre (robust z-RMS below collapse_eps)
     AND whose contrast is below the information floor is treated as "encoded
     as one data point" and dropped from the results. See drop_latent_collapse().
  C. DIAGNOSTIC METRICS — every surviving chip reports loss (VAE per-chip
     reconstruction MSE), contrast (RMS contrast std/mean), edge_regularity
     and gradient. Aggregates (mean / median / std) are printed and written.
  D. HEADLESS RUNNER —  python xenarch_mk20_script.py --run [DIR]
     trains/scores a whole folder of imagery with no Flask server and writes
     <out>_chip_metrics.csv + <out>_summary.json. Flags: --epochs --max-chips
     --tile-mode/--crop --no-baseline --out. With no --run it still serves the
     Flask app exactly like Mk19.

Key changes vs Mk14/Mk17:
  1. TRIMMED ROBUST TRAINING — after a warmup, each epoch the top-k% highest
     reconstruction-error chips are EXCLUDED from gradient updates, so anomalies
     are never absorbed into the "natural geology" baseline. This is the fix for
     the core contamination problem (the VAE previously memorized the lander).
  2. AUGMENTATION — random flips / 90-degree rotations force the VAE to learn
     geology statistics instead of memorizing individual chips.
  3. DETERMINISTIC SCORING — chips are scored through the latent mean (mu),
     no sampling noise in the ranking.
  4. PATCH-WISE MAX ERROR — reconstruction error is pooled over local windows
     and the MAX is taken, so a small artifact dominates its chip instead of
     being averaged away by 65k background pixels.
  5. LATENT MAHALANOBIS DISTANCE — robust per-dimension z-distance in latent
     space, statistics fit on inlier chips only (replaces distance-from-origin).
  6. TWO-SIDED CONTEXTUAL SCORE — detects dark compact features (shadowed
     hardware) as well as bright ones.
  7. ORIENTATION-INVARIANT EDGE REGULARITY — FFT angular-spectrum concentration
     catches straight edges at ANY angle; weighted up from 5% to 25%.
  8. ROBUST NORMALIZATION — median/MAD z-scores + sigmoid instead of min-max,
     so one extreme chip can't compress the rest of the distribution.
  9. CALIBRATED CONFIDENCE — sigmoid of the robust z of the combined score;
     stays LOW when nothing in the scene is genuinely anomalous.
 10. 50% CHIP OVERLAP — features straddling chip boundaries are no longer
     split and diluted.
 11. TRAINING FOLDER BASELINE — the VAE trains on imagery in the project's
     `training/` folder (curated natural geology; searched recursively;
     TIF/PNG/JPG/NPY). Uploaded scenes are then scored against that FIXED
     baseline: normalization statistics, the anomaly-percentile threshold,
     and confidence z-scores all come from the training distribution, so
     scene scores mean "how far from known-natural" rather than "how weird
     relative to this scene". Override the folder with XENARCH_TRAINING_DIR
     or per-job via config["training_dir"]. If the folder is missing or
     empty, the pipeline falls back to Mk19 self-supervised trimmed training.

Local dev:
    pip install -r requirements.txt
    python xenarch_mk20_script.py

Production (via gunicorn — set by Procfile automatically):
    gunicorn xenarch_mk20_script:app

Environment variables (all optional):
    PORT            — HTTP port (default 5000)
    HOST            — bind address (default 0.0.0.0)
    ALLOWED_ORIGINS — comma-separated CORS origins. Defaults to * for easy
                      first deployment.
"""

import os
import sys
import json
import base64
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import List, Dict, Optional
from io import BytesIO

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter, label
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from loguru import logger
from PIL import Image

# ── optional heavy deps ──────────────────────────────────────────────────────
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logger.warning("rasterio not available — falling back to Pillow for image I/O")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
    logger.info("PyTorch available — VAE training enabled")
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available — using numpy-only anomaly scoring")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────────────

CHIP_SIZE = 256
PROGRESS: Dict = {}
RESULTS:  Dict = {}

# ── training baseline folder ─────────────────────────────────────────────────
# The VAE trains on imagery found here (your curated "all natural geology"
# set) and uploaded scenes are scored against that baseline. Defaults to the
# `training/` folder inside the project folder (next to this script); override
# with the XENARCH_TRAINING_DIR env var or per-job with config["training_dir"].
# Subfolders are searched recursively. If missing/empty, the pipeline falls
# back to self-supervised trimmed training on the uploaded scene.
TRAINING_DIR = Path(os.environ.get(
    "XENARCH_TRAINING_DIR",
    str(Path(__file__).resolve().parent / "training data")))
TRAIN_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy"}


def list_training_images(train_dir) -> List[Path]:
    try:
        d = Path(train_dir)
        if not d.is_dir():
            return []
        return sorted(p for p in d.rglob("*")
                      if p.is_file() and p.suffix.lower() in TRAIN_EXTS)
    except Exception:
        return []


METRIC_KEYS = ["mse", "latent", "contextual", "gradient", "edge"]

# Combined-score weights. Edge regularity (straight lines at any orientation)
# is the strongest geology-vs-technology discriminator, so it carries real
# weight now. Patch-max reconstruction error remains the primary VAE signal.
COMBINED_WEIGHTS = {
    "mse":        0.30,   # patch-wise MAX reconstruction error
    "latent":     0.15,   # robust latent-space distance
    "contextual": 0.20,   # compact bright OR dark feature + texture outlier
    "gradient":   0.10,   # local gradient irregularity
    "edge":       0.25,   # orientation-invariant edge regularity
}

_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
CORS_ORIGINS = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else "*"

app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS, supports_credentials=True)

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | {level} | {message}",
)

# ─────────────────────────────────────────────────────────────────────────────
# SMALL NUMERIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def robust_z(values: np.ndarray) -> np.ndarray:
    """Median/MAD z-scores. Immune to a single extreme outlier, unlike min-max."""
    v = np.asarray(values, dtype=np.float64)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    if mad < 1e-12:
        mad = v.std() + 1e-12
    return (v - med) / mad


def robust_norm(values: np.ndarray) -> np.ndarray:
    """Squash robust z-scores through a sigmoid into (0,1)."""
    z = robust_z(values)
    return 1.0 / (1.0 + np.exp(-z / 2.0))


def local_std(a: np.ndarray, size: int = 9) -> np.ndarray:
    """Fast closed-form local std (replaces the very slow generic_filter(np.std))."""
    m  = uniform_filter(a, size=size, mode="reflect")
    m2 = uniform_filter(a * a, size=size, mode="reflect")
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def shannon_entropy(chip: np.ndarray, bins: int = 64) -> float:
    """Entropy (bits) of the chip's intensity histogram. A flat grey/white
    field lands almost entirely in one bin -> entropy near 0."""
    hist, _ = np.histogram(chip, bins=bins, range=(0.0, 1.0))
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p[p > 0] / s
    return float(-(p * np.log2(p)).sum())


def contrast_metrics(chip: np.ndarray) -> Dict[str, float]:
    """RMS contrast (std / mean) plus the 1–99 percentile dynamic range."""
    mean = float(chip.mean())
    std  = float(chip.std())
    p1, p99 = np.percentile(chip, [1, 99])
    return {
        "contrast":      float(std / (mean + 1e-6)),   # RMS / Michelson-ish
        "contrast_std":  std,
        "dyn_range":     float(p99 - p1),
        "mean_level":    mean,
    }


# Low-information gate. A chip must clear at least one real-structure test;
# otherwise it is a flat grey / blown-white / dead-black field that the VAE
# would just collapse onto the latent centroid.
LOWINFO_DEFAULTS = dict(
    min_dyn_range = 0.06,   # 1–99 pct spread on a 0..1 image
    min_entropy   = 3.2,    # bits, 64-bin histogram
    min_grad      = 0.010,  # mean gradient magnitude
    edge_thresh   = 0.05,   # abs gradient magnitude counted as an "edge" pixel
    min_edge_frac = 0.010,  # fraction of pixels above edge_thresh
    white_level   = 0.97,   # mean above this AND low range -> "too white"
    dark_level    = 0.03,   # mean below this AND low range -> "too dark"
)


def chip_information_ok(chip: np.ndarray, cfg: Optional[Dict] = None) -> Dict:
    """Decide whether a chip carries enough information to be worth modelling.

    Returns {"ok": bool, "reason": str, ...diagnostics}. `reason` is "" when ok.
    A chip is rejected only when it fails the dynamic-range / entropy test AND
    has no meaningful gradient or edge content — i.e. it really is featureless.
    """
    c = dict(LOWINFO_DEFAULTS)
    if cfg:
        c.update({k: cfg[k] for k in LOWINFO_DEFAULTS if k in cfg})

    cm = contrast_metrics(chip)
    ent = shannon_entropy(chip)

    gx = np.gradient(chip, axis=0)
    gy = np.gradient(chip, axis=1)
    gmag = np.sqrt(gx ** 2 + gy ** 2)
    grad_mean = float(gmag.mean())
    # absolute edge fraction — pixels whose gradient magnitude clears a fixed
    # threshold (a self-relative percentile would be ~10% for ANY noisy chip).
    edge_frac = float((gmag > c["edge_thresh"]).mean()) if gmag.size else 0.0

    diag = {
        "dyn_range":   cm["dyn_range"],
        "entropy":     ent,
        "grad_mean":   grad_mean,
        "edge_frac":   edge_frac,
        "mean_level":  cm["mean_level"],
        "contrast":    cm["contrast"],
    }

    has_range   = cm["dyn_range"] >= c["min_dyn_range"]
    has_entropy = ent >= c["min_entropy"]
    has_struct  = grad_mean >= c["min_grad"] or edge_frac >= c["min_edge_frac"]

    if has_struct and (has_range or has_entropy):
        return {"ok": True, "reason": "", **diag}

    if cm["mean_level"] >= c["white_level"]:
        reason = "too_white"
    elif cm["mean_level"] <= c["dark_level"]:
        reason = "too_dark"
    elif not has_struct and not has_range:
        reason = "too_flat"
    else:
        reason = "low_information"
    return {"ok": False, "reason": reason, **diag}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  IMAGE LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_image_as_array(path: str) -> np.ndarray:
    if str(path).lower().endswith(".npy"):
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        lo, hi = np.percentile(arr, [1, 99])
        if hi - lo < 1e-8:
            return np.clip(arr, 0.0, 1.0)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    if HAS_RASTERIO:
        try:
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
            lo, hi = np.percentile(arr, [1, 99])
            arr = np.clip(arr, lo, hi)
            arr = (arr - lo) / (hi - lo + 1e-8)
            return arr
        except Exception:
            pass
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CHIP EXTRACTION  (now with overlap so features aren't split at seams)
# ─────────────────────────────────────────────────────────────────────────────

def extract_chips(image_path: str, output_dir: str,
                  chip_size: int = CHIP_SIZE,
                  overlap: float = 0.5,
                  max_chips: int = 500) -> List[Dict]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    arr = load_image_as_array(image_path)
    h, w = arr.shape
    stride = max(int(round(chip_size * (1.0 - overlap))), 16)
    chips, chip_id = [], 0
    source_stem = Path(image_path).stem

    for y in range(0, h - chip_size + 1, stride):
        for x in range(0, w - chip_size + 1, stride):
            if chip_id >= max_chips:
                break
            chip = arr[y:y + chip_size, x:x + chip_size]
            if chip.std() < 0.005:          # keep near-flat chips only out if truly empty
                continue
            chip_filename = f"{source_stem}_chip_{chip_id:04d}.npy"
            chip_path = output_path / chip_filename
            np.save(str(chip_path), chip.astype(np.float32))
            chips.append({
                "chip_id":   chip_id,
                "chip_path": str(chip_path),
                "center_x":  x + chip_size // 2,
                "center_y":  y + chip_size // 2,
                "source":    Path(image_path).name,
            })
            chip_id += 1
        if chip_id >= max_chips:
            break

    logger.info(f"  Extracted {chip_id} chips from {Path(image_path).name} (stride={stride})")
    return chips


# ─────────────────────────────────────────────────────────────────────────────
# 3.  NUMPY MULTI-METRIC SCORER  (also used as fallback when torch is absent)
# ─────────────────────────────────────────────────────────────────────────────

class NumpyAnomalyScorer:
    """Hand-crafted metrics. When torch is unavailable this is the whole scorer;
    when torch is available it supplies the contextual/gradient/edge metrics."""

    def __init__(self, reference_chips: Optional[List[np.ndarray]] = None):
        self.f_med = None
        self.f_mad = None
        if reference_chips:
            feats = np.stack([self._features(c) for c in reference_chips[:400]])
            self.f_med = np.median(feats, axis=0)
            self.f_mad = np.median(np.abs(feats - self.f_med), axis=0) * 1.4826 + 1e-8

    # -- simple statistical fingerprint of a chip, for the fallback latent score
    @staticmethod
    def _features(chip: np.ndarray) -> np.ndarray:
        gx = np.gradient(chip, axis=0)
        gy = np.gradient(chip, axis=1)
        gm = np.sqrt(gx ** 2 + gy ** 2)
        ls = local_std(chip, 9)
        return np.array([
            chip.mean(), chip.std(),
            gm.mean(), gm.std(),
            ls.mean(), ls.std(),
            np.percentile(chip, 99) - np.percentile(chip, 1),
        ], dtype=np.float64)

    def latent_score(self, chip: np.ndarray) -> float:
        """Fallback 'latent' distance: robust z-distance of the chip's statistical
        fingerprint from the population. (Torch path replaces this with the VAE's
        latent Mahalanobis distance.)"""
        if self.f_med is None:
            return float(chip.std())
        z = (self._features(chip) - self.f_med) / self.f_mad
        return float(np.sqrt(np.mean(z ** 2)))

    def mse_score(self, chip: np.ndarray, patch: int = 32) -> float:
        """Fallback reconstruction-style score: PATCH-WISE MAX deviation from a
        smooth background model, so a small artifact isn't averaged away."""
        smooth = gaussian_filter(chip, sigma=8)
        resid = (chip - smooth) ** 2
        pooled = uniform_filter(resid, size=patch, mode="reflect")
        return float(pooled.max())

    def contextual_score(self, chip: np.ndarray) -> Dict:
        """Compact locally-deviant region, BRIGHT or DARK (shadowed hardware),
        plus texture-outlier fraction."""
        mean_b = chip.mean()
        std_b  = chip.std() + 1e-8

        ls = local_std(chip, 9)
        tex_mean = ls.mean()
        tex_std  = ls.std() + 1e-8
        texture_a = float((np.abs(ls - tex_mean) > 2 * tex_std).mean())

        best_score, best_bbox = 0.0, None
        for mask in (chip > mean_b + 2 * std_b,      # bright anomaly
                     chip < mean_b - 2 * std_b):     # dark anomaly / shadow
            if mask.sum() < 5:
                continue
            dev = abs(float(chip[mask].mean()) - mean_b)
            intensity_a = min(dev / (std_b * 3), 1.0)

            labeled, n_regions = label(mask)
            comp_a, bbox = 0.0, None
            for rid in range(1, n_regions + 1):
                region = labeled == rid
                size = int(region.sum())
                if size < 4:
                    continue
                ys, xs = np.where(region)
                y1, y2 = int(ys.min()), int(ys.max())
                x1, x2 = int(xs.min()), int(xs.max())
                bbox_area = (y2 - y1 + 1) * (x2 - x1 + 1)
                compactness = size / (bbox_area + 1e-8)
                if compactness > comp_a:
                    comp_a = compactness
                    bbox = [y1 / chip.shape[0], x1 / chip.shape[1],
                            y2 / chip.shape[0], x2 / chip.shape[1]]
            score = 0.35 * intensity_a + 0.35 * texture_a + 0.30 * comp_a
            if score > best_score:
                best_score, best_bbox = score, bbox

        return {"score": float(best_score), "bbox": best_bbox}

    def gradient_score(self, chip: np.ndarray) -> float:
        gx = np.gradient(chip, axis=0)
        gy = np.gradient(chip, axis=1)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        local_g = gaussian_filter(grad_mag, sigma=4)
        return float(np.abs(grad_mag - local_g).mean())

    def edge_regularity_score(self, chip: np.ndarray) -> float:
        """Orientation-invariant straight-edge detector.
        (a) alignment of strong edges along rows/cols (axis-aligned lines) and
        (b) FFT angular-spectrum concentration: natural terrain has an
            isotropic spectrum; straight edges at ANY angle concentrate energy
            in a narrow angular band."""
        h, w = chip.shape
        gx = np.gradient(chip, axis=0)
        gy = np.gradient(chip, axis=1)
        edge_str = np.sqrt(gx ** 2 + gy ** 2)
        thresh = np.percentile(edge_str, 90)
        strong = edge_str > thresh
        line_a = 0.0
        if strong.sum() >= 10:
            row_align = strong.sum(axis=1).max() / (strong.sum() + 1e-8)
            col_align = strong.sum(axis=0).max() / (strong.sum() + 1e-8)
            line_a = float(max(row_align, col_align))

        # FFT angular concentration
        win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
        spec = np.abs(np.fft.fftshift(np.fft.fft2((chip - chip.mean()) * win)))
        cy, cx = h // 2, w // 2
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot(yy - cy, xx - cx)
        theta = np.mod(np.arctan2(yy - cy, xx - cx), np.pi)
        sel = (r > 4) & (r < min(h, w) // 2)
        nbins = 36
        hist, _ = np.histogram(theta[sel], bins=nbins, range=(0, np.pi),
                               weights=spec[sel])
        total = hist.sum() + 1e-9
        conc = hist.max() / total
        # uniform spectrum -> conc = 1/nbins; strong linear feature -> conc >> 1/nbins
        fft_a = float(np.clip((conc - 1.0 / nbins) / (0.25 - 1.0 / nbins), 0.0, 1.0))

        return float(0.5 * line_a + 0.5 * fft_a)

    def score(self, chip: np.ndarray) -> Dict:
        ctx = self.contextual_score(chip)
        return {
            "mse":          self.mse_score(chip),
            "latent":       self.latent_score(chip),
            "contextual":   ctx["score"],
            "gradient":     self.gradient_score(chip),
            "edge":         self.edge_regularity_score(chip),
            "feature_bbox": ctx["bbox"],
        }

    @staticmethod
    def fit_norm_stats(score_list: List[Dict]) -> Dict[str, tuple]:
        """Fit robust (median/MAD) normalization statistics — typically on the
        TRAINING baseline, so scene chips are measured against 'natural'."""
        stats = {}
        for k in METRIC_KEYS:
            v = np.array([s[k] for s in score_list], dtype=np.float64)
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med)) * 1.4826)
            if mad < 1e-12:
                mad = float(v.std()) + 1e-12
            stats[k] = (med, mad)
        return stats

    @staticmethod
    def apply_norm(score_list: List[Dict], stats: Dict[str, tuple]) -> List[Dict]:
        """Robust z + sigmoid normalization with externally supplied statistics;
        a single extreme chip can't compress the rest the way min-max did."""
        result = []
        for s in score_list:
            s_out = dict(s)
            combined = 0.0
            for k in METRIC_KEYS:
                med, mad = stats[k]
                z = (s[k] - med) / mad
                n = float(1.0 / (1.0 + np.exp(-z / 2.0)))
                s_out[f"{k}_norm"] = n
                combined += COMBINED_WEIGHTS[k] * n
            s_out["combined"] = float(combined)
            result.append(s_out)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TORCH VAE
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TORCH:

    class StableConvolutionalVAE(nn.Module):
        def __init__(self, latent_dim=56, input_size=256):
            super().__init__()
            self.latent_dim = latent_dim
            self.input_size = input_size
            final_size = input_size // 16
            self.final_size = final_size
            self.encoder_conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1),  nn.BatchNorm2d(32, eps=1e-3),  nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64, eps=1e-3),  nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128, eps=1e-3), nn.ReLU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256, eps=1e-3), nn.ReLU(),
            )
            self.fc_mu     = nn.Linear(256 * final_size * final_size, latent_dim)
            self.fc_logvar = nn.Linear(256 * final_size * final_size, latent_dim)
            nn.init.xavier_uniform_(self.fc_mu.weight, gain=0.01)
            nn.init.xavier_uniform_(self.fc_logvar.weight, gain=0.01)
            self.decoder_input = nn.Linear(latent_dim, 256 * final_size * final_size)
            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(128, eps=1e-3), nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(64, eps=1e-3), nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(32, eps=1e-3), nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid(),
            )

        def encode(self, x):
            h = torch.flatten(self.encoder_conv(x), 1)
            mu, logvar = self.fc_mu(h), self.fc_logvar(h)
            return mu, torch.clamp(logvar, -10, 10)

        @staticmethod
        def reparameterize(mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std

        def decode(self, z):
            h = self.decoder_input(z).view(-1, 256, self.final_size, self.final_size)
            return self.decoder_conv(h)

        def forward(self, x):
            mu, logvar = self.encode(x)
            return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    class TorchDataset(Dataset):
        """augment=True applies random flips / 90-degree rotations so the VAE
        learns geology statistics rather than memorizing individual chips."""
        def __init__(self, chip_paths, augment=False):
            self.paths = chip_paths
            self.augment = augment

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            arr = np.load(self.paths[idx]).astype(np.float32)
            if self.augment:
                k = np.random.randint(4)
                if k:
                    arr = np.rot90(arr, k)
                if np.random.rand() < 0.5:
                    arr = np.fliplr(arr)
                arr = np.ascontiguousarray(arr)
            return torch.from_numpy(arr[np.newaxis]), self.paths[idx]

    def stable_vae_loss(recon, x, mu, logvar, beta=0.01, kl_weight=1.0):
        # beta raised from 0.001 -> 0.01: a bit more regularization discourages
        # memorization of rare (anomalous) chips.
        mse = F.mse_loss(recon, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + beta * kl_weight * kld, mse, kld

    def per_chip_recon_mse(model, chip_paths, device, batch_size) -> np.ndarray:
        """Deterministic (mu-path) whole-chip MSE, used to select training inliers."""
        model.eval()
        out = []
        loader = DataLoader(TorchDataset(chip_paths, augment=False),
                            batch_size=batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(device)
                mu, _ = model.encode(imgs)
                recon = model.decode(mu)
                m = torch.mean((imgs - recon) ** 2, dim=[1, 2, 3])
                out.extend(m.cpu().numpy().tolist())
        model.train()
        return np.array(out)

    def patchwise_max_error(imgs, recon, patch=32) -> "torch.Tensor":
        """Local mean of squared error, then MAX over patches. A small artifact
        now dominates its chip's score instead of being averaged away."""
        err = (imgs - recon) ** 2
        pooled = F.avg_pool2d(err, kernel_size=patch, stride=max(patch // 2, 1))
        return pooled.flatten(1).max(dim=1).values


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CONFIDENCE  (calibrated: low when nothing is truly anomalous)
# ─────────────────────────────────────────────────────────────────────────────

def compute_confidence(scored_chips: List[Dict],
                       ref_combined: Optional[List[float]] = None) -> List[Dict]:
    """Sigmoid of the robust z of the combined score, centered at z=2.
    A chip 2 robust sigmas above the reference median -> 0.5; 4 sigmas -> ~0.88.
    When a training baseline exists, z is measured against the BASELINE
    distribution ('how far from natural geology?'); otherwise against the
    scene itself. Either way, the top chip is NOT automatically 'confident' —
    if the scene is all natural, everything scores low."""
    comb = np.array([c["combined"] for c in scored_chips], dtype=np.float64)
    base = (np.asarray(ref_combined, dtype=np.float64)
            if ref_combined is not None else comb)
    med = np.median(base)
    mad = np.median(np.abs(base - med)) * 1.4826
    if mad < 1e-12:
        mad = base.std() + 1e-12
    z = (comb - med) / mad
    conf = 1.0 / (1.0 + np.exp(-(z - 2.0)))
    out = []
    for i, c in enumerate(scored_chips):
        c2 = dict(c)
        c2["confidence"] = float(np.clip(conf[i], 0.0, 1.0))
        out.append(c2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6.  THUMBNAIL
# ─────────────────────────────────────────────────────────────────────────────

def chip_to_b64(chip_path: str, size: int = 256) -> str:
    try:
        arr = np.load(chip_path)
    except Exception:
        arr = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").resize((size, size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(job_id: str, image_paths: List[str], config: Dict, tmp_dir: str):
    def log(msg, level="info"):
        entry = {"t": time.strftime("%H:%M:%S.") + f"{int(time.time()*1000)%1000:03d}",
                 "msg": msg, "level": level}
        PROGRESS[job_id]["logs"].append(entry)
        getattr(logger, level if level in ("info", "warning", "error") else "info")(msg)

    def set_step(n, pct):
        PROGRESS[job_id]["step"] = n
        PROGRESS[job_id]["pct"] = pct

    try:
        PROGRESS[job_id] = {"step": 0, "pct": 0, "logs": [], "done": False, "error": None}

        chip_dir   = os.path.join(tmp_dir, "chips")
        chip_size  = int(config.get("chip_size", CHIP_SIZE))
        overlap    = float(config.get("overlap", 0.5))
        percentile = float(config.get("percentile", 92))
        epochs     = int(config.get("epochs", 20))
        latent_dim = int(config.get("latent_dim", 56))
        batch_size = int(config.get("batch_size", 4))
        lr         = float(config.get("lr", 0.0005))
        warmup     = int(config.get("warmup_epochs", 3))
        trim_frac  = float(np.clip(config.get("trim_frac", 0.08), 0.0, 0.4))

        # ── 1. chips ────────────────────────────────────────────────────────
        set_step(1, 5)
        log(f"Chip extractor: chip_size={chip_size}, overlap={overlap:.0%}")
        all_chips = []
        for path in image_paths:
            log(f"Extracting chips from {Path(path).name}…")
            chips = extract_chips(path, chip_dir, chip_size=chip_size, overlap=overlap)
            all_chips.extend(chips)
            log(f"  → {len(chips)} chips extracted", "success" if chips else "warning")

        if not all_chips:
            raise ValueError("No chips extracted — check image dimensions (need ≥ chip size).")

        log(f"Total chips: {len(all_chips)}", "success")
        set_step(1, 15)

        chip_paths = [c["chip_path"] for c in all_chips]

        # ── 1b. training baseline from the project's training/ folder ───────
        # If the folder exists and contains imagery, the VAE trains ONLY on it
        # (your curated "all natural geology" set) and the uploaded scenes are
        # scored against that fixed baseline. If it's missing or empty, we fall
        # back to the Mk19 self-supervised trimmed training on the scene chips.
        train_dir  = Path(config.get("training_dir") or TRAINING_DIR)
        train_imgs = list_training_images(train_dir)
        max_train  = int(config.get("max_train_chips", 800))
        train_chip_paths: List[str] = []
        if train_imgs:
            log(f"Training baseline: {len(train_imgs)} image(s) in {train_dir}")
            train_chip_dir = os.path.join(tmp_dir, "train_chips")
            budget = max(max_train // len(train_imgs), 20)
            for p in train_imgs:
                tchips = extract_chips(str(p), train_chip_dir, chip_size=chip_size,
                                       overlap=overlap, max_chips=budget)
                train_chip_paths.extend(c["chip_path"] for c in tchips)
                if len(train_chip_paths) >= max_train:
                    train_chip_paths = train_chip_paths[:max_train]
                    break
            log(f"Baseline chips: {len(train_chip_paths)}",
                "success" if train_chip_paths else "warning")
        use_baseline = len(train_chip_paths) >= 8
        if not use_baseline:
            log(f"No usable training data in {train_dir} — falling back to "
                "self-supervised (trimmed) training on the uploaded scene.", "warning")
            train_chip_paths = list(chip_paths)

        # ── 2. model ────────────────────────────────────────────────────────
        set_step(2, 18)
        if HAS_TORCH:
            model_used = ("VAE-Mk20 (PyTorch, natural baseline)" if use_baseline
                          else "VAE-Mk20 (PyTorch, trimmed self)")
        else:
            model_used = "NumPy robust scorer"
        log(f"Fitting {model_used}: train={len(train_chip_paths)} chips, "
            f"score={len(chip_paths)} chips")

        # numpy scorer statistics are fit on the TRAINING distribution
        _ref_sample = [np.load(p) for p in train_chip_paths[:300]]
        np_scorer = NumpyAnomalyScorer(_ref_sample)
        del _ref_sample

        if HAS_TORCH:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model  = StableConvolutionalVAE(latent_dim=latent_dim, input_size=chip_size).to(device)
            optim_ = torch.optim.Adam(model.parameters(), lr=lr)
            sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_, factor=0.5, patience=2)

            train_paths = list(train_chip_paths)
            keep_idx = np.arange(len(train_chip_paths))

            for epoch in range(epochs):
                # TRIMMED TRAINING: after warmup, drop the top trim_frac highest
                # reconstruction-error chips from this epoch's gradient updates.
                # Even a curated training folder can contain accidental
                # contamination; suspected outliers never get learned in.
                if epoch >= warmup and trim_frac > 0 and len(train_chip_paths) > 10:
                    losses = per_chip_recon_mse(model, train_chip_paths, device, batch_size)
                    n_keep = max(int(len(train_chip_paths) * (1.0 - trim_frac)), 8)
                    keep_idx = np.argsort(losses)[:n_keep]
                    train_paths = [train_chip_paths[i] for i in keep_idx]
                    if epoch == warmup:
                        log(f"Trimmed training active: excluding top {trim_frac:.0%} "
                            f"({len(train_chip_paths) - n_keep} chips) from gradient updates")

                loader = DataLoader(TorchDataset(train_paths, augment=True),
                                    batch_size=batch_size, shuffle=True, num_workers=0)
                model.train()
                kl_w = min(1.0, (epoch + 1) / max(warmup, 1))
                total_loss = 0.0
                nan_hit = False
                for imgs, _ in loader:
                    imgs = imgs.to(device)
                    optim_.zero_grad()
                    recon, mu, logvar = model(imgs)
                    loss, mse, kld = stable_vae_loss(recon, imgs, mu, logvar, kl_weight=kl_w)
                    if torch.isnan(loss):
                        log(f"NaN at epoch {epoch+1}!", "error")
                        nan_hit = True
                        break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim_.step()
                    total_loss += loss.item()
                if nan_hit:
                    break
                avg = total_loss / max(len(train_paths), 1)
                sched.step(avg)
                log(f"Epoch {epoch+1}/{epochs} [KL={kl_w:.2f}] "
                    f"loss={avg:.1f} (train set: {len(train_paths)} chips)")
                set_step(2, 18 + int((epoch + 1) / epochs * 34))

            # ── deterministic scoring through mu (no sampling noise) ────────
            model.eval()
            patch = max(chip_size // 8, 16)

            def encode_and_score(paths):
                errs, mu_list = [], []
                loader = DataLoader(TorchDataset(paths, augment=False),
                                    batch_size=batch_size, shuffle=False, num_workers=0)
                with torch.no_grad():
                    for imgs, _ in loader:
                        imgs = imgs.to(device)
                        mu, _ = model.encode(imgs)
                        recon = model.decode(mu)
                        pm = patchwise_max_error(imgs, recon, patch=patch)
                        errs.extend(pm.cpu().numpy().tolist())
                        mu_list.append(mu.cpu().numpy())
                return np.array(errs), np.concatenate(mu_list, axis=0)

            log("Scoring baseline chips (deterministic mu-path)…")
            ref_mse, ref_mus = encode_and_score(train_chip_paths)
            if use_baseline:
                log("Scoring scene chips against baseline…")
                scn_mse, scn_mus = encode_and_score(chip_paths)
            else:
                scn_mse, scn_mus = ref_mse, ref_mus   # scene == training set

            # latent distance: robust per-dim z fit on INLIER baseline chips
            # only, so anomalies can't contaminate the reference distribution
            inlier = ref_mus[keep_idx] if len(keep_idx) >= 8 else ref_mus
            l_med = np.median(inlier, axis=0)
            l_mad = np.median(np.abs(inlier - l_med), axis=0) * 1.4826 + 1e-8

            def latent_dist(mus):
                z = (mus - l_med) / l_mad
                return np.sqrt(np.mean(z ** 2, axis=1))

            def build_raw(paths, mses, lats, want_bbox):
                out = []
                for i, p in enumerate(paths):
                    arr = np.load(p)
                    ctx = np_scorer.contextual_score(arr)
                    out.append({
                        "mse":          float(mses[i]),
                        "latent":       float(lats[i]),
                        "contextual":   ctx["score"],
                        "gradient":     np_scorer.gradient_score(arr),
                        "edge":         np_scorer.edge_regularity_score(arr),
                        "feature_bbox": ctx["bbox"] if want_bbox else None,
                    })
                return out

            raw_scores = build_raw(chip_paths, scn_mse, latent_dist(scn_mus), True)
            ref_raw = (build_raw(train_chip_paths, ref_mse, latent_dist(ref_mus), False)
                       if use_baseline else None)
        else:
            raw_scores = [np_scorer.score(np.load(p)) for p in chip_paths]
            ref_raw = ([np_scorer.score(np.load(p)) for p in train_chip_paths]
                       if use_baseline else None)

        # ── 3. normalize ────────────────────────────────────────────────────
        # Normalization statistics come from the TRAINING baseline when one is
        # available, so scene scores are absolute ("how far from natural?")
        # rather than relative to whatever happens to be in this scene.
        set_step(3, 58)
        log("Robust-normalising scores (median/MAD, "
            + ("stats from training baseline)…" if use_baseline else "self-referenced)…"))
        stats  = NumpyAnomalyScorer.fit_norm_stats(ref_raw if use_baseline else raw_scores)
        scored = NumpyAnomalyScorer.apply_norm(raw_scores, stats)
        ref_combined = None
        if use_baseline:
            ref_scored   = NumpyAnomalyScorer.apply_norm(ref_raw, stats)
            ref_combined = [r["combined"] for r in ref_scored]
        set_step(3, 70)

        # ── 4. rank ─────────────────────────────────────────────────────────
        set_step(4, 72)
        scored = compute_confidence(scored, ref_combined=ref_combined)
        thr_src = ref_combined if use_baseline else [s["combined"] for s in scored]
        threshold = np.percentile(thr_src, percentile)
        for i, s in enumerate(scored):
            s.update(all_chips[i])
            s["is_anomaly"] = bool(s["combined"] > threshold)
        scored.sort(key=lambda x: x["combined"], reverse=True)
        n_anomaly = sum(1 for s in scored if s["is_anomaly"])
        n_high    = sum(1 for s in scored if s["confidence"] > 0.8)
        log(f"Anomalies: {n_anomaly}  |  High-conf >0.8: {n_high}", "success")
        set_step(4, 86)

        # ── 5. package ──────────────────────────────────────────────────────
        set_step(5, 88)
        log("Generating thumbnails…")
        top_n = min(12, len(scored))
        results_out = []
        for rank, s in enumerate(scored[:top_n], 1):
            results_out.append({
                "rank":        rank,
                "chipName":    Path(s["chip_path"]).stem,
                "confidence":  round(s["confidence"], 4),
                "score":       round(s["combined"], 4),
                "source":      s.get("source", ""),
                "imgDataURI":  chip_to_b64(s["chip_path"]),
                "featureBbox": s.get("feature_bbox"),
                "metrics": {k: round(s.get(f"{k}_norm", 0), 3) for k in METRIC_KEYS},
            })
        log(f"Rank 1: {results_out[0]['chipName']}  conf={results_out[0]['confidence']:.4f}",
            "success")
        set_step(5, 100)
        log("Analysis complete ✓", "success")

        RESULTS[job_id] = {
            "summary": {
                "total_chips": len(scored),
                "n_anomalies": n_anomaly,
                "n_high_conf": n_high,
                "top_conf":    round(results_out[0]["confidence"], 4) if results_out else 0,
                "model_used":  model_used,
                "baseline":    ("training folder" if use_baseline else "self-supervised"),
                "baseline_chips": len(train_chip_paths) if use_baseline else 0,
            },
            "detections": results_out,
            "csv_rows": [
                {k: s[k] for k in ("chipName", "confidence", "score", "source", "metrics")}
                for s in results_out
            ],
        }
        PROGRESS[job_id]["done"] = True

    except Exception as exc:
        logger.error(traceback.format_exc())
        PROGRESS[job_id]["error"] = str(exc)
        PROGRESS[job_id]["done"] = True


# ─────────────────────────────────────────────────────────────────────────────
# 7b.  HEADLESS DATASET RUNNER  (python xenarch_mk20_script.py --run DIR)
# ─────────────────────────────────────────────────────────────────────────────

def _iter_dataset_images(root: Path) -> List[Path]:
    return [p for p in sorted(Path(root).rglob("*"))
            if p.is_file() and p.suffix.lower() in TRAIN_EXTS]


def _load_full_chip(path: Path, chip_size: int, mode: str) -> np.ndarray:
    """One (chip_size, chip_size) float array per source file.
    mode='tile'  -> area-resize the whole image to chip_size (keeps full context)
    mode='crop'  -> centre-crop chip_size at native resolution (keeps detail)."""
    arr = load_image_as_array(str(path))
    h, w = arr.shape
    if mode == "crop" and h >= chip_size and w >= chip_size:
        y0 = (h - chip_size) // 2
        x0 = (w - chip_size) // 2
        return np.ascontiguousarray(arr[y0:y0 + chip_size, x0:x0 + chip_size], dtype=np.float32)
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")
    img = img.resize((chip_size, chip_size), Image.LANCZOS)
    return (np.asarray(img, dtype=np.float32) / 255.0)


def run_dataset(dataset_dir: str,
                out_prefix: str = "xenarch_mk20",
                chip_size: int = CHIP_SIZE,
                mode: str = "tile",
                overlap: float = 0.5,
                epochs: int = 12,
                batch_size: int = 8,
                lr: float = 5e-4,
                latent_dim: int = 56,
                warmup: int = 3,
                trim_frac: float = 0.08,
                max_chips: int = 900,
                percentile: float = 92.0,
                collapse_eps: float = 0.15,
                lowinfo_cfg: Optional[Dict] = None,
                seed: int = 0) -> Dict:
    """Train the Mk20 VAE on `dataset_dir` (self-supervised, trimmed) and report
    per-chip loss / contrast / edge-regularity / gradient, dropping
    low-information and latent-collapsed chips. Returns the summary dict and
    writes <out_prefix>_chip_metrics.csv and <out_prefix>_summary.json."""
    import csv as _csv
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)

    t_start = time.time()
    root = Path(dataset_dir)
    files = _iter_dataset_images(root)
    if not files:
        raise SystemExit(f"no images ({sorted(TRAIN_EXTS)}) under {root}")
    logger.info(f"Mk20 dataset run: {len(files)} source file(s) under {root}")
    logger.info(f"  mode={mode}  chip_size={chip_size}  epochs={epochs}  "
                f"batch={batch_size}  max_chips={max_chips}")

    tmp_dir = tempfile.mkdtemp(prefix="xenarch_mk20_")
    chip_dir = Path(tmp_dir) / "chips"
    chip_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. build chips + low-information gate ────────────────────────────────
    kept: List[Dict] = []
    dropped: List[Dict] = []
    lowinfo_cfg = lowinfo_cfg or {}

    def _ids(f: Path):
        """(root-relative source path, group) — group is the first sub-folder
        under the dataset root (here: the PDS product id), so per-group
        aggregates don't collide on the shared 'tile_NNN.png' file names."""
        try:
            rel = f.relative_to(root)
        except ValueError:
            rel = Path(f.name)
        src = str(rel).replace("\\", "/")
        grp = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        return src, grp

    def _register(arr: np.ndarray, source: str, group: str, sub_id: int):
        info = chip_information_ok(arr, lowinfo_cfg)
        rec = {"source": source, "group": group, "sub_id": sub_id,
               "dyn_range": round(info["dyn_range"], 5),
               "entropy": round(info["entropy"], 4),
               "grad_mean": round(info["grad_mean"], 6),
               "edge_frac": round(info["edge_frac"], 5),
               "mean_level": round(info["mean_level"], 5)}
        if not info["ok"]:
            rec["drop_reason"] = info["reason"]
            dropped.append(rec)
            return
        safe = source.replace("/", "__").replace(".", "_")
        cp = chip_dir / f"{safe}_{sub_id:04d}.npy"
        np.save(str(cp), arr.astype(np.float32))
        rec["chip_path"] = str(cp)
        kept.append(rec)

    for f in files:
        if len(kept) >= max_chips:
            break
        src, grp = _ids(f)
        try:
            if mode in ("tile", "crop"):
                arr = _load_full_chip(f, chip_size, mode)
                if arr.std() < 1e-4:
                    dropped.append({"source": src, "group": grp, "sub_id": 0,
                                    "drop_reason": "empty", "dyn_range": 0.0,
                                    "entropy": 0.0, "grad_mean": 0.0,
                                    "edge_frac": 0.0, "mean_level": float(arr.mean())})
                    continue
                _register(arr, src, grp, 0)
            else:  # 'extract' — sliding-window chips like the web pipeline
                full = load_image_as_array(str(f))
                h, w = full.shape
                stride = max(int(round(chip_size * (1 - overlap))), 16)
                sid = 0
                for y in range(0, h - chip_size + 1, stride):
                    for x in range(0, w - chip_size + 1, stride):
                        if len(kept) >= max_chips:
                            break
                        _register(full[y:y + chip_size, x:x + chip_size], src, grp, sid)
                        sid += 1
        except Exception as exc:
            logger.warning(f"  skip {src}: {exc}")

    n_lowinfo = len(dropped)
    logger.info(f"  chips kept after low-info gate: {len(kept)}  "
                f"(dropped {n_lowinfo})")
    if len(kept) < 8:
        raise SystemExit(f"only {len(kept)} usable chips — nothing to model")

    chip_paths = [k["chip_path"] for k in kept]

    # ── 2. train (trimmed self-supervised) or numpy fallback ────────────────
    np_scorer = NumpyAnomalyScorer([np.load(p) for p in chip_paths[:300]])
    training_curve: List[float] = []
    keep_idx = np.arange(len(chip_paths))

    if HAS_TORCH:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = StableConvolutionalVAE(latent_dim=latent_dim, input_size=chip_size).to(device)
        optim_ = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_, factor=0.5, patience=2)
        train_paths = list(chip_paths)
        for epoch in range(epochs):
            if epoch >= warmup and trim_frac > 0 and len(chip_paths) > 10:
                losses = per_chip_recon_mse(model, chip_paths, device, batch_size)
                n_keep = max(int(len(chip_paths) * (1 - trim_frac)), 8)
                keep_idx = np.argsort(losses)[:n_keep]
                train_paths = [chip_paths[i] for i in keep_idx]
            loader = DataLoader(TorchDataset(train_paths, augment=True),
                                batch_size=batch_size, shuffle=True, num_workers=0)
            model.train()
            kl_w = min(1.0, (epoch + 1) / max(warmup, 1))
            tot = 0.0
            nb = 0
            for imgs, _ in loader:
                imgs = imgs.to(device)
                optim_.zero_grad()
                recon, mu, logvar = model(imgs)
                loss, mse, kld = stable_vae_loss(recon, imgs, mu, logvar, kl_weight=kl_w)
                if torch.isnan(loss):
                    logger.error(f"NaN at epoch {epoch+1}")
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim_.step()
                tot += loss.item()
                nb += 1
            avg = tot / max(nb, 1)
            training_curve.append(avg)
            sched.step(avg)
            logger.info(f"  epoch {epoch+1}/{epochs} [KL={kl_w:.2f}] loss={avg:.1f} "
                        f"(train {len(train_paths)} chips)")

        model.eval()
        patch = max(chip_size // 8, 16)
        errs_patch, recon_mse, mus = [], [], []
        loader = DataLoader(TorchDataset(chip_paths, augment=False),
                            batch_size=batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(device)
                mu, _ = model.encode(imgs)
                recon = model.decode(mu)
                errs_patch.extend(patchwise_max_error(imgs, recon, patch=patch).cpu().numpy().tolist())
                recon_mse.extend(torch.mean((imgs - recon) ** 2, dim=[1, 2, 3]).cpu().numpy().tolist())
                mus.append(mu.cpu().numpy())
        mus = np.concatenate(mus, axis=0)
        recon_mse = np.asarray(recon_mse)
        errs_patch = np.asarray(errs_patch)
        model_used = "VAE-Mk20 (PyTorch, trimmed self-supervised)"
    else:
        mus = np.stack([np_scorer._features(np.load(p)) for p in chip_paths])
        recon_mse = np.asarray([np_scorer.mse_score(np.load(p)) for p in chip_paths])
        errs_patch = recon_mse.copy()
        model_used = "NumPy robust scorer (no torch)"

    # ── 3. latent-collapse filter ──────────────────────────────────────────
    inlier = mus[keep_idx] if len(keep_idx) >= 8 else mus
    l_med = np.median(inlier, axis=0)
    l_mad = np.median(np.abs(inlier - l_med), axis=0) * 1.4826 + 1e-8
    latent_z = np.sqrt(np.mean(((mus - l_med) / l_mad) ** 2, axis=1))

    collapse_mask = np.zeros(len(kept), dtype=bool)
    for i, k in enumerate(kept):
        contrast_low = k["dyn_range"] < LOWINFO_DEFAULTS["min_dyn_range"] * 1.5
        if latent_z[i] < collapse_eps and contrast_low:
            collapse_mask[i] = True
            d = dict(k)
            d["drop_reason"] = "latent_collapse"
            d["latent_z"] = round(float(latent_z[i]), 5)
            dropped.append(d)
    n_collapse = int(collapse_mask.sum())
    logger.info(f"  latent-collapse drops: {n_collapse}")

    # ── 4. diagnostic metrics for the survivors ───────────────────────────
    rows: List[Dict] = []
    for i, k in enumerate(kept):
        if collapse_mask[i]:
            continue
        arr = np.load(k["chip_path"])
        cm = contrast_metrics(arr)
        rows.append({
            "source":          k["source"],
            "group":           k["group"],
            "sub_id":          k["sub_id"],
            "loss":            float(recon_mse[i]),          # VAE per-chip recon MSE
            "patch_max_err":   float(errs_patch[i]),
            "contrast":        cm["contrast"],
            "contrast_std":    cm["contrast_std"],
            "dyn_range":       cm["dyn_range"],
            "mean_level":      cm["mean_level"],
            "edge_regularity": float(np_scorer.edge_regularity_score(arr)),
            "gradient":        float(np_scorer.gradient_score(arr)),
            "entropy":         float(k["entropy"]),
            "latent_z":        float(latent_z[i]),
        })

    if not rows:
        raise SystemExit("every chip was dropped — loosen the thresholds")

    def _agg(key):
        v = np.asarray([r[key] for r in rows], dtype=np.float64)
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}

    metric_summary = {m: _agg(m) for m in
                      ("loss", "contrast", "edge_regularity", "gradient")}

    # per-group (here: per PDS product) means
    per_group: Dict[str, list] = {}
    for r in rows:
        per_group.setdefault(r["group"], []).append(r)
    per_group_summary = {
        g: {m: round(float(np.mean([r[m] for r in rs])), 6)
            for m in ("loss", "contrast", "edge_regularity", "gradient")}
        | {"n_chips": len(rs)}
        for g, rs in sorted(per_group.items())
    }
    drop_by_group: Dict[str, int] = {}
    for d in dropped:
        g = d.get("group", "(root)")
        drop_by_group[g] = drop_by_group.get(g, 0) + 1

    drop_reasons: Dict[str, int] = {}
    for d in dropped:
        drop_reasons[d["drop_reason"]] = drop_reasons.get(d["drop_reason"], 0) + 1

    summary = {
        "dataset_dir":     str(root),
        "model_used":      model_used,
        "mode":            mode,
        "chip_size":       chip_size,
        "epochs":          epochs,
        "elapsed_sec":     round(time.time() - t_start, 1),
        "n_source_files":  len(files),
        "n_chips_built":   len(kept) + n_lowinfo,
        "n_chips_scored":  len(rows),
        "n_dropped_total": len(dropped),
        "drop_reasons":    drop_reasons,
        "final_train_loss": training_curve[-1] if training_curve else None,
        "training_curve":  [round(x, 2) for x in training_curve],
        "metric_summary":  metric_summary,
        "per_group":       per_group_summary,
        "drop_by_group":   drop_by_group,
    }

    csv_path = f"{out_prefix}_chip_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        cols = ["source", "group", "sub_id", "loss", "patch_max_err", "contrast",
                "contrast_std", "dyn_range", "mean_level", "edge_regularity",
                "gradient", "entropy", "latent_z"]
        w = _csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
        # dropped chips appended with their reason for a full audit trail
        w.writerow({c: "" for c in cols})
        dw = _csv.DictWriter(fh, fieldnames=["source", "group", "sub_id", "drop_reason",
                                             "dyn_range", "entropy", "grad_mean",
                                             "edge_frac", "mean_level"])
        dw.writeheader()
        for d in dropped:
            dw.writerow({k: d.get(k, "") for k in
                         ("source", "group", "sub_id", "drop_reason", "dyn_range",
                          "entropy", "grad_mean", "edge_frac", "mean_level")})

    json_path = f"{out_prefix}_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # ── console report ────────────────────────────────────────────────────
    bar = "-" * 66
    print("\n" + bar)
    print(f" XENARCH Mk20 | dataset run | {root}")
    print(bar)
    print(f" model            : {model_used}")
    print(f" source files     : {len(files)}")
    print(f" chips built      : {len(kept) + n_lowinfo}")
    print(f" chips scored     : {len(rows)}")
    print(f" dropped          : {len(dropped)}  {drop_reasons}")
    if training_curve:
        print(f" train loss       : {training_curve[0]:.1f} -> {training_curve[-1]:.1f} "
              f"({epochs} epochs)")
    print(bar)
    print(f" {'metric':<16}{'mean':>12}{'median':>12}{'std':>12}{'min':>10}{'max':>10}")
    for m in ("loss", "contrast", "edge_regularity", "gradient"):
        a = metric_summary[m]
        print(f" {m:<16}{a['mean']:>12.5f}{a['median']:>12.5f}{a['std']:>12.5f}"
              f"{a['min']:>10.4f}{a['max']:>10.4f}")
    if len(per_group_summary) > 1:
        print(bar)
        print(f" per-group means"
              f"{'loss':>16}{'contrast':>12}{'edge_reg':>12}{'gradient':>12}{'n':>6}")
        for g, a in per_group_summary.items():
            print(f"  {g[:24]:<24}{a['loss']:>14.5f}{a['contrast']:>12.4f}"
                  f"{a['edge_regularity']:>12.4f}{a['gradient']:>12.4f}{a['n_chips']:>6}")
    print(bar)
    print(f" wrote {csv_path}")
    print(f" wrote {json_path}")
    print(bar + "\n")
    return summary


def _cli():
    import argparse
    ap = argparse.ArgumentParser(
        description="Xenarch Mk20 — headless dataset runner (omit --run to serve the Flask app)")
    ap.add_argument("--run", nargs="?", const=str(TRAINING_DIR / "img curated dataset"),
                    metavar="DIR",
                    help="folder of imagery to train/score (default: "
                         "'training data/img curated dataset')")
    ap.add_argument("--out", default="xenarch_mk20", help="output file prefix")
    ap.add_argument("--mode", choices=["tile", "crop", "extract"], default="tile",
                    help="tile: resize each file to one chip; crop: centre-crop; "
                         "extract: sliding-window chips")
    ap.add_argument("--chip-size", type=int, default=CHIP_SIZE)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--latent-dim", type=int, default=56)
    ap.add_argument("--max-chips", type=int, default=900)
    ap.add_argument("--trim-frac", type=float, default=0.08)
    ap.add_argument("--collapse-eps", type=float, default=0.15)
    ap.add_argument("--percentile", type=float, default=92.0)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 8.  EMBEDDED FRONTEND  (served at GET /)
# ─────────────────────────────────────────────────────────────────────────────

FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>XENARCH · Planetary Technosignature Detection · Mk20</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700&display=swap" rel="stylesheet" />
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #04060d;
    --surface:   #090e1a;
    --panel:     #0d1526;
    --border:    #1a2744;
    --accent:    #00e5ff;
    --accent2:   #ff4f00;
    --dim:       #3a5080;
    --text:      #c8daf5;
    --textlo:    #4a6080;
    --mono:      'Share Tech Mono', monospace;
    --sans:      'Barlow Condensed', sans-serif;
    --glow:      0 0 18px rgba(0,229,255,.35);
    --glow2:     0 0 18px rgba(255,79,0,.35);
  }

  html, body { height: 100%; background: var(--bg); color: var(--text); font-family: var(--sans); }

  body::before {
    content: '';
    position: fixed; inset: 0; z-index: 9999; pointer-events: none;
    background: repeating-linear-gradient(0deg, transparent, transparent 3px,
      rgba(0,0,0,.09) 3px, rgba(0,0,0,.09) 4px);
  }

  .shell { max-width: 1280px; margin: 0 auto; padding: 0 24px 80px; }

  header {
    display: flex; align-items: center; gap: 20px;
    padding: 32px 0 24px; border-bottom: 1px solid var(--border); margin-bottom: 36px;
  }
  .logo-mark {
    width: 44px; height: 44px; border: 2px solid var(--accent); border-radius: 4px;
    display: grid; place-items: center; box-shadow: var(--glow); position: relative; flex-shrink: 0;
  }
  .logo-mark::after {
    content: ''; position: absolute; inset: 5px; border: 1px solid var(--accent);
    border-radius: 2px; opacity: .5;
  }
  .logo-cross {
    width: 16px; height: 16px;
    background: linear-gradient(var(--accent), var(--accent)) 50% 0/2px 100%,
                linear-gradient(var(--accent), var(--accent)) 0 50%/100% 2px;
    background-color: transparent;
  }
  .logo-text h1 {
    font-family: var(--mono); font-size: 22px; letter-spacing: .18em;
    color: var(--accent); text-shadow: var(--glow);
  }
  .logo-text p {
    font-size: 11px; letter-spacing: .25em; color: var(--dim);
    text-transform: uppercase; margin-top: 2px;
  }
  .header-badge {
    margin-left: auto; font-family: var(--mono); font-size: 10px; color: var(--dim);
    letter-spacing: .1em; text-align: right; line-height: 1.8;
  }
  .header-badge span { color: var(--accent); }

  .main-grid { display: grid; grid-template-columns: 320px 1fr; gap: 24px; }
  @media (max-width: 860px) { .main-grid { grid-template-columns: 1fr; } }

  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 24px; }
  .panel-title {
    font-family: var(--mono); font-size: 11px; letter-spacing: .18em; color: var(--dim);
    text-transform: uppercase; margin-bottom: 18px; padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
  }

  #dropzone {
    border: 2px dashed var(--border); border-radius: 6px; padding: 36px 16px;
    text-align: center; cursor: pointer; transition: border-color .2s, background .2s; position: relative;
  }
  #dropzone:hover, #dropzone.drag { border-color: var(--accent); background: rgba(0,229,255,.04); }
  #dropzone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }
  #dropzone .dz-icon { font-size: 28px; margin-bottom: 10px; opacity: .5; }
  #dropzone .dz-label { font-size: 13px; color: var(--textlo); line-height: 1.6; }
  #dropzone .dz-label span { color: var(--accent); }

  #file-list { margin-top: 12px; display: flex; flex-direction: column; gap: 6px; }
  .file-tag {
    background: rgba(0,229,255,.07); border: 1px solid var(--border); border-radius: 4px;
    padding: 6px 10px; font-size: 11px; font-family: var(--mono); color: var(--text);
    display: flex; align-items: center; justify-content: space-between;
  }
  .file-tag button { background: none; border: none; color: var(--dim); cursor: pointer; font-size: 13px; }
  .file-tag button:hover { color: var(--accent2); }

  .param-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px; }
  .param-block label {
    display: block; font-size: 10px; letter-spacing: .12em; color: var(--dim);
    margin-bottom: 5px; text-transform: uppercase;
  }
  .param-block input, .param-block select {
    width: 100%; background: var(--surface); border: 1px solid var(--border); border-radius: 4px;
    color: var(--text); font-family: var(--mono); font-size: 13px; padding: 7px 10px;
    outline: none; transition: border-color .15s;
  }
  .param-block input:focus, .param-block select:focus { border-color: var(--accent); }

  #run-btn {
    margin-top: 22px; width: 100%; padding: 13px; background: transparent;
    border: 2px solid var(--accent); border-radius: 4px; color: var(--accent);
    font-family: var(--mono); font-size: 14px; letter-spacing: .18em; cursor: pointer;
    text-transform: uppercase; box-shadow: var(--glow);
    transition: background .2s, color .2s, box-shadow .2s; position: relative; overflow: hidden;
  }
  #run-btn:hover:not(:disabled) {
    background: var(--accent); color: var(--bg); box-shadow: 0 0 28px rgba(0,229,255,.6);
  }
  #run-btn:disabled { opacity: .4; cursor: not-allowed; }

  .right-col { display: flex; flex-direction: column; gap: 20px; }

  #progress-panel { display: none; }
  .progress-steps {
    display: flex; gap: 0; margin-bottom: 20px; border: 1px solid var(--border);
    border-radius: 4px; overflow: hidden;
  }
  .step-item {
    flex: 1; padding: 10px 6px; text-align: center; font-size: 10px; letter-spacing: .08em;
    text-transform: uppercase; color: var(--dim); border-right: 1px solid var(--border);
    transition: background .3s, color .3s;
  }
  .step-item:last-child { border-right: none; }
  .step-item.active { background: rgba(0,229,255,.12); color: var(--accent); }
  .step-item.done   { background: rgba(0,229,255,.06); color: var(--text); }

  .prog-bar-outer { height: 4px; background: var(--border); border-radius: 2px; margin-bottom: 16px; overflow: hidden; }
  .prog-bar-inner {
    height: 100%; background: var(--accent); border-radius: 2px;
    box-shadow: var(--glow); width: 0%; transition: width .4s ease;
  }

  #log-box {
    background: var(--surface); border: 1px solid var(--border); border-radius: 4px;
    padding: 12px 14px; height: 160px; overflow-y: auto; font-family: var(--mono);
    font-size: 11px; line-height: 1.9; color: var(--textlo);
  }
  #log-box .log-ok   { color: #40e090; }
  #log-box .log-warn { color: #ffb347; }
  #log-box .log-err  { color: #ff4f4f; }

  #results-panel { display: none; }
  .summary-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }
  .summary-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 4px;
    padding: 14px; text-align: center;
  }
  .summary-card .sc-val {
    font-family: var(--mono); font-size: 26px; color: var(--accent);
    text-shadow: var(--glow); line-height: 1;
  }
  .summary-card .sc-label {
    font-size: 10px; letter-spacing: .12em; color: var(--dim);
    text-transform: uppercase; margin-top: 5px;
  }

  #detections-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 14px; }

  .det-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
    overflow: hidden; cursor: pointer; transition: border-color .2s, transform .15s; position: relative;
  }
  .det-card:hover { border-color: var(--accent); transform: translateY(-2px); }
  .det-card.rank1 { border-color: var(--accent2); box-shadow: var(--glow2); }
  .det-card.rank1 .det-rank { background: var(--accent2); color: #fff; }

  .det-img-wrap { position: relative; aspect-ratio: 1; background: #000; overflow: hidden; }
  .det-img-wrap img {
    width: 100%; height: 100%; object-fit: cover; display: block;
    filter: brightness(.9) contrast(1.1); image-rendering: pixelated;
  }
  .det-bbox {
    position: absolute; border: 2px solid var(--accent2);
    box-shadow: 0 0 8px rgba(255,79,0,.6); pointer-events: none;
  }
  .rank1 .det-bbox { border-color: #ff0; box-shadow: 0 0 10px rgba(255,255,0,.7); }

  .det-rank {
    position: absolute; top: 6px; left: 6px; background: rgba(0,0,0,.75);
    border: 1px solid var(--border); border-radius: 3px; font-family: var(--mono);
    font-size: 10px; padding: 2px 6px; color: var(--dim); backdrop-filter: blur(4px);
  }

  .det-info { padding: 10px 12px; }
  .det-chip-name {
    font-family: var(--mono); font-size: 10px; color: var(--textlo); margin-bottom: 6px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .det-conf-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .det-conf-label { font-size: 10px; color: var(--dim); letter-spacing: .08em; text-transform: uppercase; }
  .det-conf-val   { font-family: var(--mono); font-size: 14px; color: var(--accent); margin-left: auto; }
  .rank1 .det-conf-val { color: var(--accent2); }

  .det-bar-outer { height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; }
  .det-bar-inner { height: 100%; background: var(--accent); border-radius: 2px; }
  .rank1 .det-bar-inner { background: var(--accent2); }

  .det-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 3px 8px; margin-top: 8px; }
  .det-metric { font-size: 9px; color: var(--textlo); display: flex; justify-content: space-between; }
  .det-metric span { color: var(--text); font-family: var(--mono); }

  #modal-overlay {
    display: none; position: fixed; inset: 0; z-index: 1000;
    background: rgba(4,6,13,.88); backdrop-filter: blur(6px);
    align-items: center; justify-content: center;
  }
  #modal-overlay.open { display: flex; }
  #modal-box {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    max-width: 640px; width: 92%; padding: 28px; position: relative;
  }
  #modal-close {
    position: absolute; top: 14px; right: 16px; background: none; border: none;
    color: var(--dim); font-size: 20px; cursor: pointer;
  }
  #modal-close:hover { color: var(--accent2); }
  #modal-img {
    width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 4px;
    image-rendering: pixelated; margin-bottom: 18px;
  }
  #modal-title { font-family: var(--mono); font-size: 14px; color: var(--accent); margin-bottom: 16px; }
  .modal-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .mm-row { display: flex; flex-direction: column; gap: 4px; }
  .mm-label { font-size: 10px; letter-spacing: .12em; color: var(--dim); text-transform: uppercase; }
  .mm-bar-outer { height: 4px; background: var(--border); border-radius: 2px; }
  .mm-bar-inner { height: 100%; background: var(--accent); border-radius: 2px; }
  .mm-val { font-family: var(--mono); font-size: 12px; color: var(--text); }

  #export-btn {
    margin-top: 18px; padding: 10px 20px; background: transparent;
    border: 1px solid var(--border); border-radius: 4px; color: var(--dim);
    font-family: var(--mono); font-size: 12px; letter-spacing: .12em; cursor: pointer;
    transition: border-color .2s, color .2s;
  }
  #export-btn:hover { border-color: var(--accent); color: var(--accent); }

  .error-banner {
    background: rgba(255,79,79,.1); border: 1px solid #ff4f4f; border-radius: 4px;
    padding: 12px 16px; font-family: var(--mono); font-size: 12px; color: #ff8a8a; margin-top: 12px;
  }

  @keyframes scanY {
    0%,100% { transform: translateY(0); opacity: 1; }
    50%      { transform: translateY(28px); opacity: .4; }
  }
  .logo-cross { animation: scanY 3s ease-in-out infinite; }
</style>
</head>
<body>
<div class="shell">

  <!-- HEADER -->
  <header>
    <div class="logo-mark"><div class="logo-cross"></div></div>
    <div class="logo-text">
      <h1>XENARCH</h1>
      <p>Planetary Surface Technosignature Detection — Mk20</p>
    </div>
    <div class="header-badge" id="status-badge">
      BACKEND<br>
      <span id="badge-torch">LOADING…</span><br>
      <span id="badge-train"></span>
    </div>
  </header>

  <div class="main-grid">

    <!-- LEFT: Controls -->
    <aside>
      <div class="panel">
        <div class="panel-title">// Input Images</div>

        <div id="dropzone">
          <input type="file" id="file-input" multiple accept=".tif,.tiff,.png,.jpg,.jpeg,.npy" />
          <div class="dz-icon">⊕</div>
          <div class="dz-label">Drop planetary imagery here<br><span>or click to browse</span><br>TIF · PNG · JPG · NPY</div>
        </div>
        <div id="file-list"></div>

        <div class="panel-title" style="margin-top:24px">// Detection Parameters</div>
        <div class="param-grid">
          <div class="param-block">
            <label>Chip Size (px)</label>
            <select id="p-chip">
              <option value="128">128</option>
              <option value="256" selected>256</option>
              <option value="512">512</option>
            </select>
          </div>
          <div class="param-block">
            <label>Chip Overlap</label>
            <select id="p-overlap">
              <option value="0">0%</option>
              <option value="25">25%</option>
              <option value="50" selected>50%</option>
            </select>
          </div>
          <div class="param-block">
            <label>Anomaly %ile</label>
            <input type="number" id="p-pct" value="92" min="50" max="99" step="1" />
          </div>
          <div class="param-block">
            <label>Train Trim %</label>
            <input type="number" id="p-trim" value="8" min="0" max="25" step="1" />
          </div>
          <div class="param-block">
            <label>Epochs</label>
            <input type="number" id="p-epochs" value="20" min="1" max="50" />
          </div>
          <div class="param-block">
            <label>Latent Dim</label>
            <input type="number" id="p-latent" value="56" min="8" max="256" />
          </div>
          <div class="param-block">
            <label>Batch Size</label>
            <input type="number" id="p-batch" value="4" min="1" max="32" />
          </div>
          <div class="param-block">
            <label>Learn Rate</label>
            <input type="number" id="p-lr" value="0.0005" min="0.00001" max="0.01" step="0.00001" />
          </div>
        </div>

        <button id="run-btn" disabled>▶ RUN ANALYSIS</button>
        <div id="error-area"></div>
      </div>
    </aside>

    <!-- RIGHT: Progress + Results -->
    <div class="right-col">

      <!-- PROGRESS -->
      <div class="panel" id="progress-panel">
        <div class="panel-title">// Pipeline Status</div>
        <div class="progress-steps">
          <div class="step-item" id="s1">1 · Chip Extract</div>
          <div class="step-item" id="s2">2 · Robust Train</div>
          <div class="step-item" id="s3">3 · Normalise</div>
          <div class="step-item" id="s4">4 · Rank</div>
          <div class="step-item" id="s5">5 · Package</div>
        </div>
        <div class="prog-bar-outer"><div class="prog-bar-inner" id="prog-bar"></div></div>
        <div id="log-box"></div>
      </div>

      <!-- RESULTS -->
      <div class="panel" id="results-panel">
        <div class="panel-title">// Detection Results</div>
        <div class="summary-row" id="summary-row"></div>
        <div id="detections-grid"></div>
        <button id="export-btn">⬇ Export CSV</button>
      </div>

    </div><!-- /right-col -->
  </div><!-- /main-grid -->
</div><!-- /shell -->

<!-- MODAL -->
<div id="modal-overlay">
  <div id="modal-box">
    <button id="modal-close">✕</button>
    <div id="modal-title"></div>
    <img id="modal-img" src="" alt="chip" />
    <div class="modal-metrics" id="modal-metrics"></div>
  </div>
</div>

<script>
/* ── API base: auto-detect same host ───────────────────────────────────── */
const API = window.location.origin;

/* ── Status badge ──────────────────────────────────────────────────────── */
async function checkStatus() {
  try {
    const r = await fetch(`${API}/api/status`);
    const d = await r.json();
    const badge = document.getElementById('badge-torch');
    badge.textContent = d.torch ? 'VAE-Mk20 (PyTorch)' : 'NumPy robust scorer';
    badge.style.color = d.torch ? 'var(--accent)' : '#ffb347';
    const tb = document.getElementById('badge-train');
    tb.textContent = d.training_images > 0
      ? `BASELINE · ${d.training_images} IMG`
      : 'BASELINE · SELF (training/ empty)';
    tb.style.color = d.training_images > 0 ? 'var(--accent)' : '#ffb347';
  } catch(e) {
    document.getElementById('badge-torch').textContent = 'OFFLINE';
  }
}
checkStatus();

/* ── File handling ─────────────────────────────────────────────────────── */
let selectedFiles = [];
const dz     = document.getElementById('dropzone');
const fi     = document.getElementById('file-input');
const fl     = document.getElementById('file-list');
const runBtn = document.getElementById('run-btn');

fi.addEventListener('change', () => addFiles(fi.files));
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('drag'); addFiles(e.dataTransfer.files); });

function addFiles(raw) {
  Array.from(raw).forEach(f => {
    if (!selectedFiles.find(x => x.name === f.name)) selectedFiles.push(f);
  });
  renderFileList();
}
function removeFile(name) {
  selectedFiles = selectedFiles.filter(f => f.name !== name);
  renderFileList();
}
function renderFileList() {
  fl.innerHTML = '';
  selectedFiles.forEach(f => {
    const tag = document.createElement('div');
    tag.className = 'file-tag';
    const kb = (f.size / 1024).toFixed(0);
    tag.innerHTML = `<span>${f.name} <span style="color:var(--textlo)">${kb}KB</span></span>
                     <button onclick="removeFile('${f.name}')">×</button>`;
    fl.appendChild(tag);
  });
  runBtn.disabled = selectedFiles.length === 0;
}

/* ── Run analysis ──────────────────────────────────────────────────────── */
let activeJobId = null;
let pollTimer   = null;
let csvData     = [];

runBtn.addEventListener('click', startAnalysis);

async function startAnalysis() {
  if (!selectedFiles.length) return;
  document.getElementById('error-area').innerHTML = '';

  const config = {
    chip_size:    +document.getElementById('p-chip').value,
    overlap:      +document.getElementById('p-overlap').value / 100,
    percentile:   +document.getElementById('p-pct').value,
    trim_frac:    +document.getElementById('p-trim').value / 100,
    epochs:       +document.getElementById('p-epochs').value,
    latent_dim:   +document.getElementById('p-latent').value,
    batch_size:   +document.getElementById('p-batch').value,
    lr:           +document.getElementById('p-lr').value,
  };

  const fd = new FormData();
  selectedFiles.forEach(f => fd.append('files[]', f));
  fd.append('config', JSON.stringify(config));

  runBtn.disabled = true;
  showProgress(true);
  resetSteps();
  document.getElementById('results-panel').style.display = 'none';
  clearLog();

  try {
    const r = await fetch(`${API}/api/analyze`, { method:'POST', body: fd });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    activeJobId = d.job_id;
    pollTimer = setInterval(pollProgress, 1000);
  } catch(e) {
    showError(e.message);
    runBtn.disabled = false;
  }
}

async function pollProgress() {
  if (!activeJobId) return;
  try {
    const r = await fetch(`${API}/api/progress/${activeJobId}`);
    const d = await r.json();

    setProgress(d.pct, d.step);
    (d.logs || []).forEach(appendLog);

    if (d.error) {
      clearInterval(pollTimer);
      showError(d.error);
      runBtn.disabled = false;
      return;
    }

    if (d.done) {
      clearInterval(pollTimer);
      await loadResults();
      runBtn.disabled = false;
    }
  } catch(e) { /* network blip, retry */ }
}

async function loadResults() {
  const r = await fetch(`${API}/api/results/${activeJobId}`);
  const d = await r.json();
  if (d.error) { showError(d.error); return; }
  csvData = d.csv_rows || [];
  renderSummary(d.summary);
  renderDetections(d.detections);
  document.getElementById('results-panel').style.display = 'block';
}

/* ── UI helpers ────────────────────────────────────────────────────────── */
function showProgress(on) {
  document.getElementById('progress-panel').style.display = on ? 'block' : 'none';
}

function resetSteps() {
  for (let i = 1; i <= 5; i++) {
    const el = document.getElementById('s' + i);
    el.classList.remove('active', 'done');
  }
  document.getElementById('prog-bar').style.width = '0%';
}

function setProgress(pct, step) {
  document.getElementById('prog-bar').style.width = pct + '%';
  for (let i = 1; i <= 5; i++) {
    const el = document.getElementById('s' + i);
    el.classList.remove('active', 'done');
    if (i < step) el.classList.add('done');
    else if (i === step) el.classList.add('active');
  }
}

const logBox = document.getElementById('log-box');
const seenLogs = new Set();
function appendLog(entry) {
  const key = entry.t + entry.msg;
  if (seenLogs.has(key)) return;
  seenLogs.add(key);
  const cls = entry.level === 'success' ? 'log-ok' : entry.level === 'error' ? 'log-err' : entry.level === 'warning' ? 'log-warn' : '';
  logBox.innerHTML += `<div class="${cls}">[${entry.t}] ${entry.msg}</div>`;
  logBox.scrollTop = logBox.scrollHeight;
}
function clearLog() { logBox.innerHTML = ''; seenLogs.clear(); }

function showError(msg) {
  document.getElementById('error-area').innerHTML = `<div class="error-banner">ERROR: ${msg}</div>`;
}

/* ── Summary cards ─────────────────────────────────────────────────────── */
function renderSummary(s) {
  const row = document.getElementById('summary-row');
  row.innerHTML = [
    { val: s.total_chips,  label: 'Total Chips' },
    { val: s.n_anomalies,  label: 'Anomalies' },
    { val: s.n_high_conf,  label: 'High Conf' },
    { val: (s.top_conf*100).toFixed(2)+'%', label: 'Top Confidence' },
  ].map(c => `
    <div class="summary-card">
      <div class="sc-val">${c.val}</div>
      <div class="sc-label">${c.label}</div>
    </div>`).join('');
}

/* ── Detection grid ────────────────────────────────────────────────────── */
function renderDetections(dets) {
  const grid = document.getElementById('detections-grid');
  grid.innerHTML = '';
  dets.forEach((d, idx) => {
    const conf = (d.confidence * 100).toFixed(2);
    const card = document.createElement('div');
    card.className = 'det-card' + (idx === 0 ? ' rank1' : '');
    card.onclick = () => openModal(d);

    let bboxHtml = '';
    if (d.featureBbox) {
      const [y1n, x1n, y2n, x2n] = d.featureBbox;
      bboxHtml = `<div class="det-bbox" style="
        top:${(y1n*100).toFixed(1)}%;
        left:${(x1n*100).toFixed(1)}%;
        width:${((x2n-x1n)*100).toFixed(1)}%;
        height:${((y2n-y1n)*100).toFixed(1)}%;
      "></div>`;
    }

    const mkeys = ['mse','latent','contextual','gradient','edge'];
    const mrows = mkeys.map(k =>
      `<div class="det-metric">${k} <span>${d.metrics[k]}</span></div>`
    ).join('');

    card.innerHTML = `
      <div class="det-img-wrap">
        <img src="${d.imgDataURI}" alt="chip" />
        ${bboxHtml}
        <div class="det-rank">#${d.rank}</div>
      </div>
      <div class="det-info">
        <div class="det-chip-name">${d.chipName}</div>
        <div class="det-conf-row">
          <div class="det-conf-label">CONF</div>
          <div class="det-conf-val">${conf}%</div>
        </div>
        <div class="det-bar-outer"><div class="det-bar-inner" style="width:${conf}%"></div></div>
        <div class="det-metrics">${mrows}</div>
      </div>`;
    grid.appendChild(card);
  });
}

/* ── Modal ─────────────────────────────────────────────────────────────── */
const overlay = document.getElementById('modal-overlay');
document.getElementById('modal-close').onclick = () => overlay.classList.remove('open');
overlay.addEventListener('click', e => { if (e.target === overlay) overlay.classList.remove('open'); });

function openModal(d) {
  document.getElementById('modal-title').textContent =
    `RANK #${d.rank}  ·  ${d.chipName}  ·  ${(d.confidence*100).toFixed(2)}% confidence`;
  document.getElementById('modal-img').src = d.imgDataURI;

  const mm = document.getElementById('modal-metrics');
  mm.innerHTML = Object.entries(d.metrics).map(([k, v]) => `
    <div class="mm-row">
      <div class="mm-label">${k}</div>
      <div class="mm-bar-outer"><div class="mm-bar-inner" style="width:${(v*100).toFixed(0)}%"></div></div>
      <div class="mm-val">${v}</div>
    </div>`).join('');

  overlay.classList.add('open');
}

/* ── CSV export ────────────────────────────────────────────────────────── */
document.getElementById('export-btn').addEventListener('click', () => {
  if (!csvData.length) return;
  const cols = ['chipName','confidence','score','source',
                'mse','latent','contextual','gradient','edge'];
  const header = cols.join(',');
  const rows = csvData.map(r => [
    r.chipName, r.confidence, r.score, r.source,
    r.metrics.mse, r.metrics.latent, r.metrics.contextual,
    r.metrics.gradient, r.metrics.edge
  ].join(','));
  const blob = new Blob([header+'\n'+rows.join('\n')], {type:'text/csv'});
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(blob), download: 'xenarch_mk20_results.csv'
  });
  a.click();
});
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# 9.  ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(FRONTEND_HTML)


@app.route("/api/status")
def api_status():
    train_imgs = list_training_images(TRAINING_DIR)
    return jsonify({"status": "ok", "version": "mk20",
                    "torch": HAS_TORCH, "rasterio": HAS_RASTERIO,
                    "training_dir": str(TRAINING_DIR),
                    "training_images": len(train_imgs)})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "files[]" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    config = {}
    if "config" in request.form:
        try:
            config = json.loads(request.form["config"])
        except Exception:
            pass
    tmp_dir  = tempfile.mkdtemp(prefix="xenarch_")
    uploaded = []
    for f in request.files.getlist("files[]"):
        dest = os.path.join(tmp_dir, os.path.basename(f.filename))
        f.save(dest)
        uploaded.append(dest)
    job_id = f"job_{int(time.time()*1000)}"
    threading.Thread(target=run_analysis,
                     args=(job_id, uploaded, config, tmp_dir), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/progress/<job_id>")
def api_progress(job_id):
    p = PROGRESS.get(job_id)
    if p is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify({"step": p["step"], "pct": p["pct"],
                    "logs": p["logs"][-50:], "done": p["done"], "error": p["error"]})


@app.route("/api/results/<job_id>")
def api_results(job_id):
    r = RESULTS.get(job_id)
    if r is None:
        p = PROGRESS.get(job_id, {})
        if p.get("error"):
            return jsonify({"error": p["error"]}), 500
        return jsonify({"error": "results not ready"}), 202
    return jsonify(r)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Headless dataset run:  python xenarch_mk20_script.py --run [DIR]
    if any(a == "--run" or a.startswith("--run=") for a in sys.argv[1:]):
        args = _cli()
        run_dataset(
            args.run, out_prefix=args.out, chip_size=args.chip_size,
            mode=args.mode, epochs=args.epochs, batch_size=args.batch_size,
            latent_dim=args.latent_dim, max_chips=args.max_chips,
            trim_frac=args.trim_frac, collapse_eps=args.collapse_eps,
            percentile=args.percentile, seed=args.seed,
        )
        sys.exit(0)

    port  = int(os.environ.get("PORT", 5000))
    host  = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info("=" * 60)
    logger.info("XENARCH Mk20 — Production Web Edition")
    logger.info(f"  PyTorch  : {'enabled' if HAS_TORCH else 'DISABLED (numpy fallback)'}")
    logger.info(f"  Rasterio : {'enabled' if HAS_RASTERIO else 'disabled (Pillow fallback)'}")
    _n_train = len(list_training_images(TRAINING_DIR))
    logger.info(f"  Training : {TRAINING_DIR} "
                f"({_n_train} image(s) found)" if _n_train
                else f"  Training : {TRAINING_DIR} (EMPTY — self-supervised fallback)")
    logger.info(f"  CORS     : {CORS_ORIGINS}")
    logger.info(f"  Serving  : http://{host}:{port}")
    logger.info("=" * 60)
    app.run(host=host, port=port, debug=debug, threaded=True)
