# 🌙 Xenarch — Planetary Technosignature Detection

*Project Prometheus*

An unsupervised machine learning system for detecting artificial structures and anomalies on
planetary surfaces. Trains a variational autoencoder exclusively on natural geology, then scores
new scenes by how far they deviate from that baseline.

**Current version: Mk19 — Production Web Edition** (`xenarch_mk19_script.py`)

---

## 🎯 Overview

By training exclusively on natural geological features, the system learns what "normal" looks
like and flags anything anomalous — including artificial structures like landing sites, rovers,
and human-made equipment.

Mk19 changed the core premise from earlier versions: instead of asking *"which chip is weird
relative to this scene?"*, it trains on a **fixed corpus of curated natural terrain** and asks
*"how far is this chip from known-natural geology?"* Scores are therefore comparable across
different scenes and runs.

### Key features

- **Fixed natural baseline** — the VAE trains on the `training data/` folder; normalization
  statistics, anomaly thresholds, and confidence z-scores all come from that distribution
- **Trimmed robust training** — after a warmup, the highest-reconstruction-error chips are
  excluded from gradient updates each epoch, so anomalies are never absorbed into the
  "natural geology" baseline
- **Five-metric ensemble** — reconstruction error, latent distance, contextual compactness,
  gradient irregularity, and orientation-invariant edge regularity
- **Calibrated confidence** — stays low when nothing in a scene is genuinely anomalous; the
  top-ranked chip is not automatically "confident"
- **Browser dashboard** — drag-and-drop upload, live pipeline logs, ranked detection grid with
  per-metric breakdowns, CSV export
- **Graceful degradation** — runs without PyTorch (NumPy scorer) or rasterio (Pillow I/O)

### Applications

- Planetary surface analysis
- Archaeological site detection
- Infrastructure monitoring on Mars/Moon missions
- Geological anomaly identification
- Change detection in satellite imagery

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+ (verified on 3.11)
- CUDA-capable GPU optional — CPU is fine, see [Performance](#-performance)

### Installation

```bash
git clone https://github.com/CSTR100/Project-Prometheus.git
cd Project-Prometheus

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run the web app

```bash
python xenarch_mk19_script.py
```

Then open <http://localhost:5000>. Drop in a scene, adjust parameters if you want, and hit
**RUN ANALYSIS**. The dashboard streams pipeline logs while it works and renders the ranked
detections when it finishes.

For production, the included `Procfile` runs it under gunicorn:

```bash
gunicorn -w 1 -k gthread --threads 8 --timeout 120 xenarch_mk19_script:app
```

> **Single worker is required.** Job progress and results live in in-process dictionaries, so a
> multi-worker deployment will return `unknown job` when a poll lands on a different worker.

### Use the pipeline as a library

`xenarch_pipeline.py` is a Flask-free, importable version for scripted evaluation:

```python
import json
from xenarch_pipeline import run_pipeline

config = json.load(open("config_iter3.json"))
results = run_pipeline("Test data/Apollo 11 landing site.png", config)

for r in results[:5]:
    print(r["rank"], r["chip_id"], r["confidence"], r["fine_bbox"])
```

---

## 📁 Project Structure

```
Project-Prometheus/
├── xenarch_mk19_script.py      # ★ Current: web app + VAE pipeline (Flask)
├── xenarch_pipeline.py         # Importable, Flask-free scoring pipeline
├── lroc_fetch.py               # LRO NAC / Chandrayaan-2 OHRC terrain harvester
├── config_iter2.json           # Alternate metric weightings for the pipeline
├── config_iter3.json           # (edge-weighted variant)
├── requirements.txt            # Python dependencies
├── Procfile                    # gunicorn entry point for PaaS deploys
├── README.md                   # This file
├── Architecture Documentation  # Detailed technical documentation
│
├── training data/              # ★ Natural terrain baseline (44 images, in repo)
│   └── dowload-img-various-api.py   # HiRISE / CTX / LRO / MOC downloader
├── Test data/                  # Validation scenes with known artifacts
├── results/                    # Generated visualizations from earlier versions
│
└── xenarch_mk3 … mk17_script.py     # Archived earlier versions
```

Chips and job output are written to a per-job temporary directory, not into the repo.

---

## 🔬 How It Works

### 1. Chip extraction

Scenes are tiled into 256×256 chips at **50% overlap**, so features straddling a chip boundary
are not split and diluted. Near-flat chips (std < 0.005) are skipped.

### 2. Baseline training

Every image in `training data/` is chipped (up to `max_train_chips`, default 800) and the VAE
trains on that corpus:

- **Augmentation** — random flips and 90° rotations, forcing the model to learn geology
  statistics rather than memorize individual chips
- **Trimmed training** — after `warmup_epochs`, the top `trim_frac` highest-error chips are
  dropped from each epoch's gradient updates, so accidental contamination in the training folder
  never gets learned in

If `training data/` is missing or empty, the pipeline falls back to self-supervised trimmed
training on the uploaded scene itself.

### 3. Scoring

Each chip is scored on five metrics, combined with these weights:

| Metric | Weight | What it measures |
|---|---|---|
| `mse` | 0.30 | Patch-wise **max** reconstruction error — a small artifact dominates its chip instead of being averaged away over 65k pixels |
| `edge` | 0.25 | Orientation-invariant edge regularity via FFT angular-spectrum concentration — catches straight edges at *any* angle |
| `contextual` | 0.20 | Compact locally-deviant region, **bright or dark** (shadowed hardware), plus texture-outlier fraction |
| `latent` | 0.15 | Robust per-dimension Mahalanobis distance in latent space, fit on inlier chips only |
| `gradient` | 0.10 | Local gradient irregularity |

Scoring runs through the latent mean (`mu`) with no sampling noise, so rankings are deterministic
given a trained model.

### 4. Normalization and confidence

Raw scores are converted to robust median/MAD z-scores and squashed through a sigmoid — one
extreme chip cannot compress the rest of the distribution the way min-max did. Statistics come
from the **training baseline**, so a score means "how far from natural geology," not "how weird
relative to this scene."

Confidence is the sigmoid of the robust z of the combined score, centered at z=2: a chip 2 robust
sigmas above the baseline median scores 0.5, and 4 sigmas scores ~0.88.

---

## 🛠️ Configuration

Parameters are set in the dashboard sidebar, or passed as JSON to `POST /api/analyze`:

| Parameter | Default | Notes |
|---|---|---|
| `chip_size` | 256 | 128, 256, or 512 (must be divisible by 16) |
| `overlap` | 0.5 | Fraction of chip size |
| `percentile` | 92 | Anomaly threshold against the baseline distribution |
| `trim_frac` | 0.08 | Fraction of highest-error chips excluded from training |
| `epochs` | 20 | |
| `latent_dim` | 56 | |
| `batch_size` | 4 | |
| `lr` | 0.0005 | |
| `warmup_epochs` | 3 | Trimming starts after this |
| `max_train_chips` | 800 | Baseline corpus size (API only) |
| `training_dir` | `training data/` | Per-job override (API only) |

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | 5000 | HTTP port |
| `HOST` | 0.0.0.0 | Bind address |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `XENARCH_TRAINING_DIR` | `training data/` | Point at a corpus outside the repo |
| `FLASK_DEBUG` | 0 | Set to 1 for debug mode |

### Tuning

- **Higher sensitivity:** lower `percentile` (88–90), lower `latent_dim` (32)
- **Higher precision:** raise `percentile` (95–97), raise `latent_dim` (64+)
- **Faster runs:** lower `max_train_chips` and `epochs` — quality degrades gracefully

---

## 🌐 API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard |
| `/api/status` | GET | Backend capabilities — torch, rasterio, training image count |
| `/api/analyze` | POST | Multipart upload: `files[]` + optional `config` JSON. Returns `{job_id}` |
| `/api/progress/<job_id>` | GET | Step, percentage, streaming logs, done/error flags |
| `/api/results/<job_id>` | GET | Summary, ranked detections with base64 thumbnails, CSV rows |

Analysis runs in a background thread; poll `/api/progress` until `done` is true.

---

## 📦 Data

### Training data

**Included in this repository** — 44 curated natural-terrain images (~24 MB) covering lunar
maria, highlands, craters, rilles, scarps, volcanic terrain, Mars cratered plains and fossae,
plus terrestrial desert analogs.

Supported formats: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, `.npy` (searched recursively).

> Store chips as PNG rather than `.npy` — a 256×256 float32 `.npy` chip is 256 KB versus 23 KB
> for the equivalent 8-bit PNG, and images are normalized to [0,1] on load anyway.

To expand the corpus:

```bash
python lroc_fetch.py --source lroc --frames 10      # LRO NAC + Ch-2 OHRC harvester
python "training data/dowload-img-various-api.py" --dataset hirise --pct 1.0
```

Both need outbound access to NASA PDS archives. `lroc_fetch.py` needs `PRADAN_USER` /
`PRADAN_PASS` for the ISRO OHRC source. Raw PDS `.IMG` frames need `rasterio` installed to be
readable, and run 250–500 MB each — keep them out of git.

### Test data

`Test data/` holds validation scenes with known artifacts (Apollo 11 landing site). These stay
in the repository as regression fixtures.

---

## 📊 Performance

Measured end-to-end on CPU (no GPU), stock parameters, 44 training images → 800 baseline chips,
20 epochs, scoring a 24-chip scene:

| Stage | Time |
|---|---|
| Chip extraction (44 baseline images + scene) | 4 s |
| VAE training (20 epochs) | 6 min 55 s |
| Scoring + normalization + packaging | 34 s |
| **Total** | **7 min 33 s** |

A CUDA GPU cuts training time roughly 10–20×. Reducing `max_train_chips` scales training time
close to linearly.

### Example run

Apollo 11 landing site, stock parameters: 24 chips scored, 10 flagged above the 92nd-percentile
baseline threshold, top confidence 67.4%. No chip exceeded the 0.8 high-confidence bar — which is
the calibration behaving as designed, not a failure: confidence is measured against the natural
baseline, so "moderately unusual" does not get promoted to "technosignature."

---

## 🕰️ Version History

| Version | Form | Key change |
|---|---|---|
| **Mk19** | Web app | Fixed training-folder baseline; trimmed robust training; patch-max error; two-sided contextual score; FFT edge regularity at 25% weight; calibrated confidence |
| Mk16–Mk17 | Web app | First Flask editions with embedded dashboard |
| Mk10–Mk15 | CLI | Multi-metric scoring, matplotlib reporting |
| Mk5–Mk9 | CLI | Variational autoencoder, probabilistic latent space |
| Mk3–Mk4 | CLI | Baseline convolutional autoencoder |

Earlier versions are retained in the repository for reference. They need extra dependencies
(matplotlib, seaborn, pandas, scikit-learn, torchvision) — see the commented section at the
bottom of `requirements.txt`.

---

## 🐛 Known Issues

- **Detections carry no scene coordinates.** The results payload and CSV export omit
  `center_x` / `center_y`, so a flagged chip cannot be mapped back to a pixel location in the
  source image without re-deriving it from the chip index.
- **No model persistence.** Every job re-chips the training folder and retrains the VAE from
  scratch; there is no checkpoint save/load.
- **Unseeded RNG.** Augmentation, weight init, and shuffling are not seeded, so identical inputs
  and config still produce slightly different rankings run to run.
- **Single-worker deployment only** (see [Quick Start](#run-the-web-app)).
- **Temporary directories are not cleaned up** after a job completes.
- Natural linear features (rilles, fault lines, scarps) can trigger the edge metric.
- Compression artifacts and scanline texture in source imagery can saturate the edge metric — if
  every chip in a scene scores >0.93 on `edge`, that metric is contributing a constant offset
  rather than discriminating.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Roadmap

- [x] Web interface for visualization
- [x] Fixed natural-terrain training baseline
- [x] Orientation-invariant edge detection
- [ ] Scene coordinates in results and CSV export
- [ ] Model checkpointing so the baseline is trained once, not per job
- [ ] Corpus manifest with checksums for reproducible training sets
- [ ] Multi-scale analysis (combine 128×128, 256×256, 512×512)
- [ ] Transfer learning from Earth satellite imagery
- [ ] Support for multi-spectral imagery

---

## 📝 Citation

```bibtex
@software{xenarch_prometheus,
  author = {Strom, Caleb},
  title  = {Xenarch: Unsupervised Planetary Technosignature Detection},
  year   = {2026},
  url    = {https://github.com/CSTR100/Project-Prometheus}
}
```

See also the Mk13 white papers included in this repository.

## 📄 License

MIT License. *(No `LICENSE` file is present in the repository yet — one should be added to make this effective.)*

## 🙏 Acknowledgments

- NASA Lunar Reconnaissance Orbiter (LRO) team for imagery
- NASA TREK for public data access
- ISRO / ISSDC PRADAN for Chandrayaan-2 data
- Apollo missions for validation data
- PyTorch, rasterio, and SciPy communities

---

⭐ Star this repository if you find it useful!

*Last updated: August 2026 · Mk19*
