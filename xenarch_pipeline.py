"""
Xenarch anomaly detection pipeline — importable, Flask-free.

Exposes run_pipeline(image_path, config) -> List[Dict] for use by the
eval harness. Logic mirrors xenarch_mk17_script.py; keeps numpy scorer
only (VAE path is optional via use_vae=True in config).
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, generic_filter, label, uniform_filter

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── defaults (match mk17) ──────────────────────────────────────────────────

DEFAULT_COMBINED_WEIGHTS: Dict[str, float] = {
    "mse": 0.60, "density": 0.30, "contextual": 0.05, "gradient": 0.05, "edge": 0.00
}
DEFAULT_CONFIDENCE_WEIGHTS: Dict[str, float] = {
    "score": 0.50, "contextual": 0.00, "mse": 0.50
}
DEFAULT_CONFIG: Dict = {
    "chip_size": 256,
    "percentile": 92,
    "epochs": 15,
    "latent_dim": 56,
    "batch_size": 4,
    "lr": 0.0005,
    "warmup_epochs": 3,
    "combined_weights": DEFAULT_COMBINED_WEIGHTS,
    "confidence_weights": DEFAULT_CONFIDENCE_WEIGHTS,
    "use_vae": False,
}


# ── image loading ──────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Return float32 [0,1] grayscale array."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
        lo, hi = np.percentile(arr, [1, 99])
        return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
    except Exception:
        pass
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


# ── chip extraction ────────────────────────────────────────────────────────

def extract_chips(
    image_path: str,
    work_dir: str,
    chip_size: int,
    max_chips: int = 600,
) -> List[Dict]:
    """Tile image into non-overlapping chips; skip near-uniform ones."""
    arr = load_image(image_path)
    h, w = arr.shape
    stem = Path(image_path).stem
    chips: List[Dict] = []
    chip_id = 0
    for y in range(0, h - chip_size + 1, chip_size):
        for x in range(0, w - chip_size + 1, chip_size):
            if chip_id >= max_chips:
                break
            chip = arr[y : y + chip_size, x : x + chip_size]
            if chip.std() < 0.01:
                continue
            fpath = os.path.join(work_dir, f"{stem}_chip_{chip_id:04d}.npy")
            np.save(fpath, chip)
            chips.append({
                "chip_id":   chip_id,
                "chip_path": fpath,
                "center_x":  x + chip_size // 2,
                "center_y":  y + chip_size // 2,
                "top_x":     x,
                "top_y":     y,
                "source":    Path(image_path).name,
            })
            chip_id += 1
        if chip_id >= max_chips:
            break
    return chips


# ── numpy multi-metric scorer ──────────────────────────────────────────────

class NumpyScorer:
    def __init__(self, reference_chips: List[np.ndarray]):
        if reference_chips:
            stacked = np.stack([c.ravel() for c in reference_chips[:200]])
            self.ref_mean = stacked.mean(axis=0)
            self.ref_std  = stacked.std(axis=0) + 1e-8
        else:
            self.ref_mean = None
            self.ref_std  = None

    def mse(self, chip: np.ndarray) -> float:
        if self.ref_mean is None:
            return float(chip.std())
        return float(np.mean((chip.ravel() - self.ref_mean) ** 2))

    def density(self, chip: np.ndarray) -> float:
        if self.ref_mean is None:
            return float(np.abs(chip - chip.mean()).mean())
        diff = chip.ravel() - self.ref_mean
        return float(np.sqrt(np.sum((diff / self.ref_std) ** 2)) / chip.size)

    def contextual(self, chip: np.ndarray) -> Tuple[float, Optional[List]]:
        m = chip.mean(); s = chip.std() + 1e-8
        bright_mask = chip > m + 2 * s
        brightness_a = 0.0
        if bright_mask.sum() >= 5:
            brightness_a = min((np.mean(chip[bright_mask]) - m) / (s * 3), 1.0)
        try:
            local_std = generic_filter(chip, np.std, size=9)
            tex_mean = local_std.mean(); tex_std = local_std.std() + 1e-8
            texture_a = float((np.abs(local_std - tex_mean) > 2 * tex_std).mean())
        except Exception:
            texture_a = 0.0
        labeled_r, n = label(bright_mask)
        comp_a = 0.0
        for rid in range(1, n + 1):
            reg = labeled_r == rid
            if reg.sum() < 4:
                continue
            ys, xs = np.where(reg)
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xs.min()), int(xs.max())
            c = reg.sum() / ((y2 - y1 + 1) * (x2 - x1 + 1) + 1e-8)
            if c > comp_a:
                comp_a = c
        score = float(0.35 * brightness_a + 0.35 * texture_a + 0.30 * comp_a)

        # Localize bbox by finding the sub-block with the strongest axis-aligned
        # edge structure — a better proxy for artificial objects than brightness.
        best_bbox = self._edge_bbox(chip)
        return score, best_bbox

    def _edge_bbox(self, chip: np.ndarray, block: int = 32) -> Optional[List]:
        """Sub-block with strongest directional edges, penalized for circular gradient arrangements."""
        h, w = chip.shape
        gx = np.gradient(chip, axis=0)
        gy = np.gradient(chip, axis=1)
        edge_map = np.sqrt(gx ** 2 + gy ** 2)
        thresh = np.percentile(edge_map, 85)
        strong = edge_map > thresh

        best_score, best_box = -1.0, None
        for y in range(0, h - block + 1, block // 2):
            for x in range(0, w - block + 1, block // 2):
                patch = strong[y : y + block, x : x + block]
                n = patch.sum()
                if n < 5:
                    continue
                patch_f = patch.astype(np.float32)
                total = float(n) + 1e-8
                row_align = patch_f.sum(axis=1).max() / total
                col_align = patch_f.sum(axis=0).max() / total

                # Gradient direction isotropy: R≈1 = directional (straight edge),
                # R≈0 = uniform across all angles (crater rim) → penalize
                angles = np.arctan2(gy[y:y+block, x:x+block][patch],
                                    gx[y:y+block, x:x+block][patch])
                R = float(np.abs(np.mean(np.exp(2j * angles))))

                reg_score = float(max(row_align, col_align)) * float(patch_f.mean()) * R
                if reg_score > best_score:
                    best_score = reg_score
                    best_box = [y / h, x / w, (y + block) / h, (x + block) / w]
        return best_box

    def gradient(self, chip: np.ndarray) -> float:
        gx = np.gradient(chip, axis=0); gy = np.gradient(chip, axis=1)
        gm = np.sqrt(gx ** 2 + gy ** 2)
        return float(np.abs(gm - gaussian_filter(gm, sigma=4)).mean())

    def edge(self, chip: np.ndarray) -> float:
        gx = np.gradient(chip, axis=0); gy = np.gradient(chip, axis=1)
        es = np.sqrt(gx ** 2 + gy ** 2)
        thresh = np.percentile(es, 90)
        strong = es > thresh
        if strong.sum() < 10:
            return 0.0
        ra = strong.sum(axis=1).max() / (strong.sum() + 1e-8)
        ca = strong.sum(axis=0).max() / (strong.sum() + 1e-8)
        return float(max(ra, ca))


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def _local_std(arr: np.ndarray, size: int = 7) -> np.ndarray:
    """Local standard deviation via E[X²]-E[X]² — no Python per-pixel loop."""
    a  = arr.astype(np.float64)
    m  = uniform_filter(a, size=size)
    m2 = uniform_filter(a * a, size=size)
    return np.sqrt(np.maximum(m2 - m * m, 0.0)).astype(np.float32)


def localize_within_chip(
    chip: np.ndarray,
    chip_top_x: int,
    chip_top_y: int,
) -> Dict:
    """
    Stage-2 localization: dense sliding-window heatmap within one chip.

    Scores every sub-patch on three axes:
      - MSE vs chip background mean  (catches brightness anomalies)
      - Local texture deviation       (catches texture outliers)
      - Directional edge regularity   (prefers straight edges, penalises craters)

    Returns fine_bbox [x1,y1,x2,y2] in image pixel coordinates and heatmap_peak.
    """
    h, w = chip.shape
    sub    = max(16, h // 8)   # sub-patch ≈ 1/8 of chip
    stride = max(4,  sub // 4) # step size gives ~(8×4)=32 grid positions per axis

    chip_mean = float(chip.mean())
    chip_var  = float(chip.var()) + 1e-8

    gx       = np.gradient(chip, axis=1)
    gy       = np.gradient(chip, axis=0)
    edge_map = np.sqrt(gx ** 2 + gy ** 2)
    lstd     = _local_std(chip, size=7)
    tex_mean = float(lstd.mean())
    tex_std  = float(lstd.std()) + 1e-8

    ny = max(1, (h - sub) // stride + 1)
    nx = max(1, (w - sub) // stride + 1)
    heat = np.zeros((ny, nx), dtype=np.float32)

    for iy in range(ny):
        for ix in range(nx):
            y0 = iy * stride
            x0 = ix * stride
            patch = chip[y0:y0+sub, x0:x0+sub]
            ep    = edge_map[y0:y0+sub, x0:x0+sub]
            lp    = lstd[y0:y0+sub, x0:x0+sub]

            mse_s = float(np.mean((patch - chip_mean) ** 2)) / chip_var
            tex_s = abs(float(lp.mean()) - tex_mean) / tex_std

            thr_e  = float(np.percentile(ep, 80))
            strong = ep > thr_e
            if strong.sum() >= 4:
                angles = np.arctan2(
                    gy[y0:y0+sub, x0:x0+sub][strong],
                    gx[y0:y0+sub, x0:x0+sub][strong],
                )
                R      = float(abs(np.mean(np.exp(2j * angles))))
                row_a  = float(strong.sum(axis=1).max()) / (float(strong.sum()) + 1e-8)
                col_a  = float(strong.sum(axis=0).max()) / (float(strong.sum()) + 1e-8)
                edge_s = max(row_a, col_a) * R
            else:
                edge_s = 0.0

            heat[iy, ix] = 0.40 * mse_s + 0.30 * tex_s + 0.30 * edge_s

    heat_s = gaussian_filter(heat.astype(np.float64), sigma=1.5).astype(np.float32)
    lo, hi = float(heat_s.min()), float(heat_s.max())
    if hi - lo < 1e-8:
        return {"fine_bbox": None, "heatmap_peak": 0.0}
    heat_n = (heat_s - lo) / (hi - lo)

    # Region ≥65% of peak AND in top 25% of heat values
    peak_val = float(heat_n.max())
    thr = max(peak_val * 0.65, float(np.percentile(heat_n, 75)))
    hot_ys, hot_xs = np.where(heat_n >= thr)
    if len(hot_ys) == 0:
        idx = np.unravel_index(int(heat_n.argmax()), heat_n.shape)
        hot_ys, hot_xs = np.array([idx[0]]), np.array([idx[1]])

    # Heatmap cell indices → chip pixel coords → image pixel coords
    cy1 = int(hot_ys.min()) * stride
    cy2 = min(h, int(hot_ys.max()) * stride + sub)
    cx1 = int(hot_xs.min()) * stride
    cx2 = min(w, int(hot_xs.max()) * stride + sub)

    return {
        "fine_bbox":    [chip_top_x + cx1, chip_top_y + cy1,
                         chip_top_x + cx2, chip_top_y + cy2],
        "heatmap_peak": peak_val,
    }


def score_chips(
    chips_arr: List[np.ndarray],
    combined_w: Dict[str, float],
    confidence_w: Dict[str, float],
    reference_chips: Optional[List[np.ndarray]] = None,
) -> List[Dict]:
    """Score and normalize a list of chips; return per-chip dicts."""
    scorer = NumpyScorer(reference_chips if reference_chips is not None else chips_arr)
    raw: List[Dict] = []
    for chip in chips_arr:
        ctx_score, bbox = scorer.contextual(chip)
        raw.append({
            "mse":          scorer.mse(chip),
            "density":      scorer.density(chip),
            "contextual":   ctx_score,
            "gradient":     scorer.gradient(chip),
            "edge":         scorer.edge(chip),
            "feature_bbox": bbox,
        })

    keys = ["mse", "density", "contextual", "gradient", "edge"]
    norms = {k: _normalize(np.array([r[k] for r in raw])) for k in keys}

    out: List[Dict] = []
    for i, r in enumerate(raw):
        r2 = dict(r)
        for k in keys:
            r2[f"{k}_norm"] = float(norms[k][i])
        r2["combined"] = float(sum(combined_w.get(k, 0.0) * norms[k][i] for k in keys))
        out.append(r2)

    # confidence
    comb_arr = np.array([r["combined"] for r in out])
    ctx_arr  = np.array([r["contextual_norm"] for r in out])
    mse_arr  = np.array([r["mse_norm"] for r in out])
    edge_arr = np.array([r["edge_norm"] for r in out])
    grad_arr = np.array([r["gradient_norm"] for r in out])
    sn = _normalize(comb_arr)
    conf = np.clip(
        confidence_w.get("score", 0.5) * sn
        + confidence_w.get("contextual", 0.3) * ctx_arr
        + confidence_w.get("mse", 0.2) * mse_arr
        + confidence_w.get("edge", 0.0) * edge_arr
        + confidence_w.get("gradient", 0.0) * grad_arr,
        0.0, 1.0,
    )
    for i, r in enumerate(out):
        r["confidence"] = float(conf[i])

    return out


# ── optional VAE path (mirrors mk17) ──────────────────────────────────────

if HAS_TORCH:
    class _VAE(nn.Module):
        def __init__(self, latent_dim: int = 56, input_size: int = 256):
            super().__init__()
            fs = input_size // 16
            self.enc = nn.Sequential(
                nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32, eps=1e-3), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64, eps=1e-3), nn.ReLU(),
                nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128, eps=1e-3), nn.ReLU(),
                nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256, eps=1e-3), nn.ReLU(),
            )
            flat = 256 * fs * fs
            self.fc_mu  = nn.Linear(flat, latent_dim)
            self.fc_lv  = nn.Linear(flat, latent_dim)
            nn.init.xavier_uniform_(self.fc_mu.weight, gain=0.01)
            nn.init.xavier_uniform_(self.fc_lv.weight, gain=0.01)
            self.dec_in = nn.Linear(latent_dim, flat)
            self.fs = fs
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.BatchNorm2d(128, eps=1e-3), nn.ReLU(),
                nn.ConvTranspose2d(128, 64,  3, 2, 1, 1), nn.BatchNorm2d(64,  eps=1e-3), nn.ReLU(),
                nn.ConvTranspose2d(64,  32,  3, 2, 1, 1), nn.BatchNorm2d(32,  eps=1e-3), nn.ReLU(),
                nn.ConvTranspose2d(32,  1,   3, 2, 1, 1), nn.Sigmoid(),
            )

        def encode(self, x):
            h = torch.flatten(self.enc(x), 1)
            return self.fc_mu(h), torch.clamp(self.fc_lv(h), -10, 10)

        def forward(self, x):
            mu, lv = self.encode(x)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            h = self.dec_in(z).view(-1, 256, self.fs, self.fs)
            return self.dec(h), mu, lv

    class _ChipDS(TorchDataset):
        def __init__(self, paths): self.paths = paths
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            return torch.from_numpy(np.load(self.paths[i]).astype(np.float32)[None]), self.paths[i]


def _run_vae(
    chip_paths: List[str],
    chips_arr: List[np.ndarray],
    cfg: Dict,
    combined_w: Dict,
    confidence_w: Dict,
    bg_paths: Optional[List[str]] = None,
) -> List[Dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chip_size = cfg.get("chip_size", 256)
    latent_dim = cfg.get("latent_dim", 56)
    epochs = cfg.get("epochs", 15)
    lr = cfg.get("lr", 0.0005)
    batch_size = cfg.get("batch_size", 4)
    warmup = cfg.get("warmup_epochs", 3)

    model = _VAE(latent_dim=latent_dim, input_size=chip_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    train_paths = bg_paths if bg_paths else chip_paths
    loader = DataLoader(_ChipDS(train_paths), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        kl_w = min(1.0, (epoch + 1) / max(warmup, 1))
        total = 0.0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            opt.zero_grad()
            recon, mu, lv = model(imgs)
            mse_loss = F.mse_loss(recon, imgs, reduction="sum")
            kld = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
            loss = mse_loss + 0.001 * kl_w * kld
            if torch.isnan(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step(total / max(len(chip_paths), 1))

    model.eval()
    raw: List[Dict] = []
    with torch.no_grad():
        for imgs, _ in DataLoader(_ChipDS(chip_paths), batch_size=batch_size, shuffle=False):
            imgs = imgs.to(device)
            recon, mu, _ = model(imgs)
            mse_v = torch.mean((imgs - recon) ** 2, dim=[1, 2, 3]).cpu().numpy()
            dist = torch.sqrt(torch.sum(mu ** 2, dim=1)).cpu().numpy()
            for j in range(len(mse_v)):
                raw.append({"mse": float(mse_v[j]),
                            "density": float(1.0 - np.exp(-dist[j] / latent_dim)),
                            "contextual": 0.0, "gradient": 0.0, "edge": 0.0,
                            "feature_bbox": None})

    scorer = NumpyScorer([])
    for i, arr in enumerate(chips_arr):
        ctx_score, bbox = scorer.contextual(arr)
        raw[i]["contextual"]   = ctx_score
        raw[i]["feature_bbox"] = bbox
        raw[i]["gradient"]     = scorer.gradient(arr)
        raw[i]["edge"]         = scorer.edge(arr)

    # normalize + confidence (same as numpy path)
    keys = ["mse", "density", "contextual", "gradient", "edge"]
    norms = {k: _normalize(np.array([r[k] for r in raw])) for k in keys}
    out: List[Dict] = []
    for i, r in enumerate(raw):
        r2 = dict(r)
        for k in keys:
            r2[f"{k}_norm"] = float(norms[k][i])
        r2["combined"] = float(sum(combined_w.get(k, 0.0) * norms[k][i] for k in keys))
        out.append(r2)
    comb_arr = np.array([r["combined"] for r in out])
    ctx_arr  = np.array([r["contextual_norm"] for r in out])
    mse_arr  = np.array([r["mse_norm"] for r in out])
    edge_arr = np.array([r["edge_norm"] for r in out])
    grad_arr = np.array([r["gradient_norm"] for r in out])
    sn = _normalize(comb_arr)
    conf = np.clip(
        confidence_w.get("score", 0.5) * sn
        + confidence_w.get("contextual", 0.3) * ctx_arr
        + confidence_w.get("mse", 0.2) * mse_arr
        + confidence_w.get("edge", 0.0) * edge_arr
        + confidence_w.get("gradient", 0.0) * grad_arr,
        0.0, 1.0,
    )
    for i, r in enumerate(out):
        r["confidence"] = float(conf[i])
    return out


# ── public API ─────────────────────────────────────────────────────────────

def load_reference_corpus(manifest_path: str) -> List[np.ndarray]:
    """Load chip arrays from a manifest.json produced by lroc_fetch.py."""
    import json
    manifest = json.loads(Path(manifest_path).read_text())
    chips = []
    for entry in manifest.get("chips", []):
        p = entry.get("chip_path", "")
        if Path(p).exists():
            chips.append(np.load(p))
    return chips


def run_pipeline(
    image_path: str,
    config: Optional[Dict] = None,
    reference_chips: Optional[List[np.ndarray]] = None,
    bg_chip_paths: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Run anomaly detection on one image.

    reference_chips: if provided, MSE/density scores are computed relative to
                     this external natural-terrain corpus rather than the image
                     itself, making artificial structures stand out more strongly.

    Returns all chips sorted by confidence (descending), each containing:
      chip_id, center_x, center_y, source, top_x, top_y,
      mse/density/contextual/gradient/edge (raw + _norm),
      combined, confidence, rank.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    chip_size   = cfg["chip_size"]
    combined_w  = cfg.get("combined_weights",  DEFAULT_COMBINED_WEIGHTS)
    confidence_w = cfg.get("confidence_weights", DEFAULT_CONFIDENCE_WEIGHTS)
    use_vae     = cfg.get("use_vae", False) and HAS_TORCH

    work_dir = tempfile.mkdtemp(prefix="xenarch_pipe_")
    try:
        chips_meta = extract_chips(image_path, work_dir, chip_size)
        if not chips_meta:
            return []

        chips_arr  = [np.load(c["chip_path"]) for c in chips_meta]
        chip_paths = [c["chip_path"] for c in chips_meta]

        if use_vae:
            scores = _run_vae(chip_paths, chips_arr, cfg, combined_w, confidence_w,
                              bg_paths=bg_chip_paths)
        else:
            scores = score_chips(chips_arr, combined_w, confidence_w,
                                 reference_chips=reference_chips)

        results: List[Dict] = []
        for meta, score in zip(chips_meta, scores):
            results.append({**meta, **score})

        results.sort(key=lambda x: x["confidence"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        # Stage 2: fine localization within top-N chips
        top_n      = cfg.get("localize_top_n", 5)
        chip_by_id = {m["chip_id"]: arr for m, arr in zip(chips_meta, chips_arr)}
        for r in results:
            r["fine_bbox"]    = None
            r["heatmap_peak"] = None
        for r in results[:top_n]:
            loc = localize_within_chip(chip_by_id[r["chip_id"]], r["top_x"], r["top_y"])
            r.update(loc)

        return results
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
