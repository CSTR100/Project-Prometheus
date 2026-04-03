"""
Xenarch Mk14 - Web Interface Edition
=====================================
Run with:
    python xenarch_mk14_web.py

Then open your browser to:
    http://localhost:5000

The server exposes:
    POST /api/analyze   — upload images, returns JSON results
    GET  /api/status    — check server health
    GET  /              — serve the frontend UI
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
from scipy.ndimage import gaussian_filter, generic_filter, label, zoom
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from loguru import logger
from PIL import Image

# ── optional heavy deps ──────────────────────────────────────────────────────
try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logger.warning("rasterio not available — falling back to Pillow for image I/O")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
    logger.info("PyTorch available — VAE training enabled")
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available — using numpy-only anomaly scoring")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

CHIP_SIZE = 256
PROGRESS: Dict = {}           # keyed by job_id
RESULTS:  Dict = {}           # keyed by job_id

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {level} | {message}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  IMAGE LOADING  (rasterio → Pillow fallback)
# ─────────────────────────────────────────────────────────────────────────────

def load_image_as_array(path: str) -> np.ndarray:
    """Returns a float32 H×W array, normalised 0–1."""
    if HAS_RASTERIO:
        try:
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
            lo, hi = np.percentile(arr, [1, 99])
            arr = np.clip(arr, lo, hi)
            arr = (arr - lo) / (hi - lo + 1e-8)
            return arr
        except Exception:
            pass  # fall through to Pillow

    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CHIP EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_chips(image_path: str, output_dir: str,
                  chip_size: int = CHIP_SIZE,
                  max_chips: int = 300) -> List[Dict]:
    """Slice a large image into chip_size × chip_size tiles."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    arr = load_image_as_array(image_path)
    h, w = arr.shape
    stride = chip_size                # no overlap for speed

    chips = []
    chip_id = 0
    source_stem = Path(image_path).stem

    for y in range(0, h - chip_size + 1, stride):
        for x in range(0, w - chip_size + 1, stride):
            if chip_id >= max_chips:
                break
            chip = arr[y:y+chip_size, x:x+chip_size]
            if chip.std() < 0.01:          # skip blank tiles
                continue

            chip_filename = f"{source_stem}_chip_{chip_id:04d}.npy"
            chip_path = output_path / chip_filename
            np.save(str(chip_path), chip)

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

    logger.info(f"  Extracted {chip_id} chips from {Path(image_path).name}")
    return chips


# ─────────────────────────────────────────────────────────────────────────────
# 3.  NUMPY-ONLY MULTI-METRIC SCORER
# ─────────────────────────────────────────────────────────────────────────────

class NumpyAnomalyScorer:
    """
    Drop-in replacement for MultiMetricAnomalyScorer that works
    without PyTorch.  Scores each chip with the same five metrics.
    """

    def __init__(self, reference_chips: List[np.ndarray]):
        """Fit a simple reference distribution from training chips."""
        if reference_chips:
            stacked = np.stack([c.ravel() for c in reference_chips[:200]])
            self.ref_mean = stacked.mean(axis=0)
            self.ref_std  = stacked.std(axis=0) + 1e-8
        else:
            self.ref_mean = None
            self.ref_std  = None

    # ── individual metrics ────────────────────────────────────────────────

    def mse_score(self, chip: np.ndarray) -> float:
        """Reconstruction proxy: deviation from reference mean."""
        if self.ref_mean is None:
            return float(chip.std())
        diff = chip.ravel() - self.ref_mean
        return float(np.mean(diff ** 2))

    def density_score(self, chip: np.ndarray) -> float:
        """Latent-density proxy: Mahalanobis-like distance."""
        if self.ref_mean is None:
            return float(np.abs(chip - chip.mean()).mean())
        diff = chip.ravel() - self.ref_mean
        return float(np.sqrt(np.sum((diff / self.ref_std) ** 2)) / chip.size)

    def contextual_score(self, chip: np.ndarray) -> Dict:
        """Brightness / compactness anomaly — returns (score, bbox)."""
        mean_b = chip.mean()
        std_b  = chip.std() + 1e-8

        # bright-pixel anomaly
        threshold    = mean_b + 2 * std_b
        bright_mask  = chip > threshold
        
        brightness_a = 0.0
        if bright_mask.sum() >= 5:
            bright_a = np.mean(chip[bright_mask])
            brightness_a = min((bright_a - mean_b) / (std_b * 3), 1.0)

        # texture anomaly
        try:
            local_std = generic_filter(chip, np.std, size=9)
            tex_mean  = local_std.mean()
            tex_std   = local_std.std() + 1e-8
            texture_a = float((np.abs(local_std - tex_mean) > 2 * tex_std).mean())
        except Exception:
            texture_a = 0.0

        # compactness anomaly & bbox tracking
        labeled, n_regions = label(bright_mask)
        comp_a = 0.0
        best_bbox = None # [y1, x1, y2, x2] in normalized 0-1
        
        for rid in range(1, n_regions + 1):
            region = labeled == rid
            size   = region.sum()
            if size < 4: continue
            
            ys, xs = np.where(region)
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xs.min()), int(xs.max())
            bbox_area = (y2-y1+1) * (x2-x1+1)
            compactness = size / (bbox_area + 1e-8)
            
            if compactness > comp_a:
                comp_a = compactness
                # normalize bbox for frontend
                best_bbox = [y1/chip.shape[0], x1/chip.shape[1], 
                             y2/chip.shape[0], x2/chip.shape[1]]

        combined_context = float(0.35 * brightness_a + 0.35 * texture_a + 0.30 * comp_a)
        return {"score": combined_context, "bbox": best_bbox}


    def gradient_score(self, chip: np.ndarray) -> float:
        """Texture-mismatch proxy via gradient magnitude variance."""
        gx = np.gradient(chip, axis=0)
        gy = np.gradient(chip, axis=1)
        grad_mag = np.sqrt(gx**2 + gy**2)
        # Anomaly = deviation of local gradient from global mean
        local_g  = gaussian_filter(grad_mag, sigma=4)
        return float(np.abs(grad_mag - local_g).mean())

    def edge_regularity_score(self, chip: np.ndarray) -> float:
        """Geometric alignment / straightness of strong edges."""
        gx = np.gradient(chip, axis=0)
        gy = np.gradient(chip, axis=1)
        edge_str = np.sqrt(gx**2 + gy**2)
        thresh   = np.percentile(edge_str, 90)
        strong   = edge_str > thresh
        if strong.sum() < 10:
            return 0.0
        row_align = strong.sum(axis=1).max() / (strong.sum() + 1e-8)
        col_align = strong.sum(axis=0).max() / (strong.sum() + 1e-8)
        return float(max(row_align, col_align))

    # ── combined ──────────────────────────────────────────────────────────

    def score(self, chip: np.ndarray) -> Dict:
        ctx_data = self.contextual_score(chip)
        raw = {
            "mse":        self.mse_score(chip),
            "density":    self.density_score(chip),
            "contextual": ctx_data["score"],
            "gradient":   self.gradient_score(chip),
            "edge":       self.edge_regularity_score(chip),
            "feature_bbox": ctx_data["bbox"] # Store for later
        }
        return raw

    def normalize_scores(self, score_list: List[Dict]) -> List[Dict]:
        """Min-max normalise each metric across the dataset."""
        keys = ["mse", "density", "contextual", "gradient", "edge"]
        arrays = {k: np.array([s[k] for s in score_list]) for k in keys}
        norms  = {}
        for k, arr in arrays.items():
            lo, hi = arr.min(), arr.max()
            norms[k] = (arr - lo) / (hi - lo + 1e-8)

        result = []
        for i, s in enumerate(score_list):
            s_out = dict(s)
            for k in keys:
                s_out[f"{k}_norm"] = float(norms[k][i])
            # Combined score (Mk13/14 weights)
            s_out["combined"] = float(
                0.30 * norms["mse"][i]        +
                0.20 * norms["density"][i]    +
                0.30 * norms["contextual"][i] +
                0.15 * norms["gradient"][i]   +
                0.05 * norms["edge"][i]
            )
            result.append(s_out)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TORCH VAE  (only compiled when torch is present)
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TORCH:
    class StableConvolutionalVAE(nn.Module):
        def __init__(self, latent_dim=56, input_size=256):
            super().__init__()
            self.latent_dim = latent_dim
            self.input_size = input_size
            final_size = input_size // 16

            self.encoder_conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32, eps=1e-3), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64, eps=1e-3), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128, eps=1e-3), nn.ReLU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256, eps=1e-3), nn.ReLU(),
            )
            self.fc_mu     = nn.Linear(256*final_size*final_size, latent_dim)
            self.fc_logvar = nn.Linear(256*final_size*final_size, latent_dim)
            nn.init.xavier_uniform_(self.fc_mu.weight, gain=0.01)
            nn.init.xavier_uniform_(self.fc_logvar.weight, gain=0.01)

            self.decoder_input = nn.Linear(latent_dim, 256*final_size*final_size)
            self.final_size = final_size

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

        def reparameterize(self, mu, logvar):
            return mu + torch.randn_like(torch.exp(0.5*logvar)) * torch.exp(0.5*logvar)

        def decode(self, z):
            h = self.decoder_input(z).view(-1, 256, self.final_size, self.final_size)
            return self.decoder_conv(h)

        def forward(self, x):
            mu, logvar = self.encode(x)
            return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    class TorchDataset(torch.utils.data.Dataset):
        def __init__(self, chip_paths):
            self.paths = chip_paths
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            arr = np.load(self.paths[idx]).astype(np.float32)[np.newaxis]
            return torch.from_numpy(arr), self.paths[idx]

    def stable_vae_loss(recon, x, mu, logvar, beta=0.001, kl_weight=1.0):
        mse = F.mse_loss(recon, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + beta * kl_weight * kld, mse, kld


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CONFIDENCE CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_confidence(scored_chips: List[Dict]) -> List[Dict]:
    """Mk14 adaptive confidence — non-clustered path (same formula as paper)."""
    scores      = np.array([c["combined"]         for c in scored_chips])
    ctx_norms   = np.array([c.get("contextual_norm", 0) for c in scored_chips])
    mse_norms   = np.array([c.get("mse_norm", 0)  for c in scored_chips])

    lo, hi = scores.min(), scores.max()
    score_norm = (scores - lo) / (hi - lo + 1e-8)

    confidence = 0.50*score_norm + 0.30*ctx_norms + 0.20*mse_norms
    confidence = np.clip(confidence, 0, 1)

    out = []
    for i, c in enumerate(scored_chips):
        c2 = dict(c)
        c2["confidence"] = float(confidence[i])
        out.append(c2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CHIP → BASE-64 PNG THUMBNAIL
# ─────────────────────────────────────────────────────────────────────────────

def chip_to_b64(chip_path: str, size: int = 256) -> str:
    """Load a .npy chip and return a base64-encoded grayscale PNG."""
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
# 7.  MAIN ANALYSIS PIPELINE  (runs in background thread)
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(job_id: str, image_paths: List[str],
                 config: Dict, tmp_dir: str):
    def log(msg, level="info"):
        entry = {"t": time.strftime("%H:%M:%S.") + f"{int(time.time()*1000)%1000:03d}",
                 "msg": msg, "level": level}
        PROGRESS[job_id]["logs"].append(entry)
        getattr(logger, level)(msg)

    def set_step(n, pct):
        PROGRESS[job_id]["step"] = n
        PROGRESS[job_id]["pct"]  = pct

    try:
        PROGRESS[job_id] = {"step": 0, "pct": 0, "logs": [], "done": False, "error": None}

        chip_dir    = os.path.join(tmp_dir, "chips")
        chip_size   = config.get("chip_size", CHIP_SIZE)
        percentile  = config.get("percentile", 92)
        epochs      = config.get("epochs", 15)
        latent_dim  = config.get("latent_dim", 56)
        batch_size  = config.get("batch_size", 4)
        lr          = config.get("lr", 0.0005)
        warmup      = config.get("warmup_epochs", 3)

        # ── Step 1: chip extraction ────────────────────────────────────────
        set_step(1, 5)
        log(f"Chip extractor: chip_size={chip_size}")
        all_chips = []
        for path in image_paths:
            log(f"Extracting chips from {Path(path).name}…")
            chips = extract_chips(path, chip_dir, chip_size=chip_size)
            all_chips.extend(chips)
            log(f"  → {len(chips)} chips extracted", "success" if chips else "warning")

        if not all_chips:
            raise ValueError("No chips extracted — check image dimensions (need ≥256×256).")

        log(f"Total chips: {len(all_chips)}", "success")
        set_step(1, 18)

        chip_paths = [c["chip_path"] for c in all_chips]
        chips_arr  = [np.load(p) for p in chip_paths]

        # ── Step 2: train / fit scorer ─────────────────────────────────────
        set_step(2, 20)
        model_used = "VAE (PyTorch)" if HAS_TORCH else "NumPy scorer"
        log(f"Fitting {model_used} on {len(all_chips)} chips…")

        if HAS_TORCH:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model  = StableConvolutionalVAE(latent_dim=latent_dim, input_size=chip_size)
            model  = model.to(device)
            optim_ = torch.optim.Adam(model.parameters(), lr=lr)
            sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_, factor=0.5, patience=2)
            ds     = TorchDataset(chip_paths)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

            best_loss = float("inf")
            for epoch in range(epochs):
                model.train()
                kl_w = min(1.0, (epoch+1)/warmup)
                total_loss, total_mse, total_kld = 0, 0, 0
                for imgs, _ in loader:
                    imgs = imgs.to(device)
                    optim_.zero_grad()
                    recon, mu, logvar = model(imgs)
                    loss, mse, kld = stable_vae_loss(recon, imgs, mu, logvar, kl_weight=kl_w)
                    if torch.isnan(loss):
                        log(f"NaN at epoch {epoch+1}!", "error")
                        break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim_.step()
                    total_loss += loss.item(); total_mse += mse.item(); total_kld += kld.item()
                avg = total_loss / max(len(ds), 1)
                sched.step(avg)
                log(f"Epoch {epoch+1}/{epochs} [KL={kl_w:.2f}] loss={avg:.1f} mse={total_mse/len(ds):.1f} kld={total_kld/len(ds):.1f}")
                set_step(2, 20 + int((epoch+1)/epochs * 35))

            # compute scores via VAE reconstruction error
            model.eval()
            log("Computing VAE reconstruction scores…")
            raw_scores = []
            with torch.no_grad():
                for imgs, _ in DataLoader(TorchDataset(chip_paths), batch_size=batch_size, shuffle=False):
                    imgs = imgs.to(device)
                    recon, mu, logvar = model(imgs)
                    mse   = torch.mean((imgs - recon)**2, dim=[1,2,3]).cpu().numpy()
                    dist  = torch.sqrt(torch.sum(mu**2, dim=1)).cpu().numpy()
                    dense = 1.0 - np.exp(-dist / latent_dim)
                    for j in range(len(mse)):
                        raw_scores.append({"mse": float(mse[j]), "density": float(dense[j]),
                                           "contextual": 0.0, "gradient": 0.0, "edge": 0.0})
            # fill contextual/gradient/edge using numpy scorer
            np_scorer = NumpyAnomalyScorer([])
            for i, arr in enumerate(chips_arr):
                ctx_data = np_scorer.contextual_score(arr)
                raw_scores[i]["contextual"]   = ctx_data["score"]
                raw_scores[i]["feature_bbox"] = ctx_data["bbox"]
                raw_scores[i]["gradient"]     = np_scorer.gradient_score(arr)
                raw_scores[i]["edge"]         = np_scorer.edge_regularity_score(arr)

        else:
            # pure numpy path
            log(f"Using numpy-only scoring (torch not available)")
            np_scorer  = NumpyAnomalyScorer(chips_arr)
            raw_scores = [np_scorer.score(arr) for arr in chips_arr]

        set_step(3, 58)

        # ── Step 3: normalise ──────────────────────────────────────────────
        log("Normalising scores across dataset…")
        scored = NumpyAnomalyScorer([]).normalize_scores(raw_scores) if not HAS_TORCH else \
                 NumpyAnomalyScorer([]).normalize_scores(raw_scores)
        # (normalize_scores is stateless so safe to call on a fresh instance)
        scorer_tmp = NumpyAnomalyScorer([])
        scored = scorer_tmp.normalize_scores(raw_scores)
        set_step(3, 72)

        # ── Step 4: rank ───────────────────────────────────────────────────
        set_step(4, 73)
        scored = compute_confidence(scored)
        threshold = np.percentile([s["combined"] for s in scored], percentile)
        for i, s in enumerate(scored):
            s.update(all_chips[i])
            s["is_anomaly"] = bool(s["combined"] > threshold)

        scored.sort(key=lambda x: x["confidence"], reverse=True)
        n_anomaly  = sum(1 for s in scored if s["is_anomaly"])
        n_high     = sum(1 for s in scored if s["confidence"] > 0.8)

        combined_arr = np.array([s["combined"] for s in scored])
        log(f"Anomalies: {n_anomaly}  |  High-conf (>0.8): {n_high}", "success")
        log(f"Score range: [{combined_arr.min():.4f}, {combined_arr.max():.4f}]", "success")
        set_step(4, 88)

        # ── Step 5: package thumbnails ─────────────────────────────────────
        set_step(5, 90)
        log("Generating thumbnails…")
        top_n = min(12, len(scored))
        results_out = []
        for rank, s in enumerate(scored[:top_n], 1):
            thumb = chip_to_b64(s["chip_path"])
            results_out.append({
                "rank":       rank,
                "chipName":   Path(s["chip_path"]).stem,
                "confidence": round(s["confidence"], 4),
                "score":      round(s["combined"], 4),
                "source":     s.get("source", ""),
                "imgDataURI": thumb,
                "featureBbox": s.get("feature_bbox"),
                "metrics": {
                    "mse":        round(s.get("mse_norm", 0), 3),
                    "density":    round(s.get("density_norm", 0), 3),
                    "contextual": round(s.get("contextual_norm", 0), 3),
                    "gradient":   round(s.get("gradient_norm", 0), 3),
                    "edge":       round(s.get("edge_norm", 0), 3),
                },
            })
        log(f"Rank 1: {results_out[0]['chipName']}  confidence={results_out[0]['confidence']:.4f}", "success")

        set_step(5, 100)
        log("Analysis complete ✓", "success")

        RESULTS[job_id] = {
            "summary": {
                "total_chips":  len(scored),
                "n_anomalies":  n_anomaly,
                "n_high_conf":  n_high,
                "top_conf":     round(results_out[0]["confidence"], 4) if results_out else 0,
                "model_used":   model_used,
            },
            "detections": results_out,
            # full CSV-ready list (top 50)
            "csv_rows": [
                {k: s[k] for k in ("chipName","confidence","score","source","metrics")}
                for s in results_out
            ]
        }
        PROGRESS[job_id]["done"]  = True

    except Exception as exc:
        err = traceback.format_exc()
        logger.error(err)
        PROGRESS[job_id]["error"] = str(exc)
        PROGRESS[job_id]["done"]  = True


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend HTML."""
    frontend = Path(__file__).parent / "xenarch_ui_connected.html"
    if frontend.exists():
        return frontend.read_text()
    return "<h1>xenarch_ui_connected.html not found</h1><p>Make sure it is in the same folder.</p>", 404


@app.route("/api/status")
def status():
    return jsonify({"status": "ok", "torch": HAS_TORCH, "rasterio": HAS_RASTERIO})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Accept multipart/form-data with:
      - files[]:  one or more image files
      - config:   JSON string with detection parameters
    Returns: { job_id: str }
    """
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
        dest = os.path.join(tmp_dir, f.filename)
        f.save(dest)
        uploaded.append(dest)

    job_id = f"job_{int(time.time()*1000)}"
    t = threading.Thread(target=run_analysis,
                         args=(job_id, uploaded, config, tmp_dir),
                         daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/progress/<job_id>")
def progress(job_id):
    p = PROGRESS.get(job_id)
    if p is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify({
        "step": p["step"],
        "pct":  p["pct"],
        "logs": p["logs"][-50:],   # last 50 log lines
        "done": p["done"],
        "error": p["error"],
    })


@app.route("/api/results/<job_id>")
def results(job_id):
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
    port = int(os.environ.get("PORT", 5000))
    logger.info("=" * 60)
    logger.info("XENARCH Mk14 — Web Interface Edition")
    logger.info(f"  PyTorch  : {'enabled' if HAS_TORCH else 'DISABLED (numpy fallback)'}")
    logger.info(f"  Rasterio : {'enabled' if HAS_RASTERIO else 'disabled (Pillow fallback)'}")
    logger.info(f"  Serving  : http://localhost:{port}")
    logger.info("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)