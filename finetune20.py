"""
Finetuning20 - Hyperparameter search comparison for the Xenarch Mk20 VAE
=========================================================================
Compares four ways of picking the VAE's hyperparameters (learning rate,
latent dimension, batch size, trim fraction, KL beta) on the curated-dataset
chips:

  1. GRADIENT - no search at all: Mk20's fixed default hyperparameters,
                plain gradient-descent training. The baseline every other
                method is measured against.
  2. RANDOM   - random search: N configs sampled uniformly from the space,
                each trained for the full epoch budget.
  3. OPTUNA   - Bayesian search (Optuna's TPE sampler), N trials, full
                epoch budget each.
  4. HALVING  - successive halving (Optuna's SuccessiveHalvingPruner):
                same TPE search, but a trial whose validation loss looks
                bad after 1-2 epochs is PRUNED before spending the rest of
                the epoch budget on it -> same trial count, less compute.

Data handling:
  - The strip's LAST tile in every original satellite product is always
    dropped up front (not just filtered statistically): the download
    pipeline zero-pads the final block to a full square, so that tile is
    mostly black padding, never real geology. See drop_last_tile_per_group().
  - The remaining chips still pass the Mk20 low-information gate (flat
    grey/white chips with no structure) on top of that.
  - Chips are split THREE ways: train (gradient updates) / val (trial
    selection + successive-halving pruning signal) / test (held out,
    touched only for the final reported numbers).

Robustness:
  - The whole comparison (data split + all 4 methods) is repeated across
    several distinct seeds (default 3: 17, 53, 91 - not a 0/1/2 counter).
    The reported table is MEAN +/- STD across those seed-runs, using each
    run's best-by-validation trial evaluated on that run's held-out test
    chips.

Outputs:
  <out>_trials.csv          every trial, every method, every seed
  <out>_comparison.json     per-method mean/std summary + winner
  <out>_comparison.png      bar charts (mean +/- std) of test loss / contrast
                            / edge_regularity / gradient, one bar per method
  <out>_loss_curves.png     validation-loss-vs-epoch curves for the best
                            trial of each method, one line per seed

Usage:
    python finetuning20.py [--dataset DIR] [--trials 6] [--epochs 4]
                            [--max-chips 220] [--chip-size 128]
                            [--val-frac 0.2] [--test-frac 0.2]
                            [--seeds 17,53,91] [--out finetuning20]
"""

import os
import re
import sys
import csv
import json
import time
import tempfile
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

# ── load the Mk20 pipeline as a library. Importing it only builds the Flask
#    app object (app.run() is guarded by `if __name__ == "__main__"` in that
#    file) so nothing is served here. ─────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "xenarch_mk20_script", str(HERE / "xenarch_mk20_script.py"))
mk20 = importlib.util.module_from_spec(_spec)
sys.modules["xenarch_mk20_script"] = mk20
_spec.loader.exec_module(mk20)

if not mk20.HAS_TORCH:
    raise SystemExit("finetuning20 needs PyTorch (xenarch_mk20_script reports HAS_TORCH=False)")

import torch
from torch.utils.data import DataLoader
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner, NopPruner

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METHODS = ["gradient", "random", "optuna", "halving"]
DEFAULT_PARAMS = dict(lr=5e-4, latent_dim=56, batch_size=8, trim_frac=0.08, beta=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA
# ─────────────────────────────────────────────────────────────────────────────

def drop_last_tile_per_group(files: List[Path], root: Path, log) -> List[Path]:
    """Drop the highest-index tile_NNN.png in every per-product sub-folder.
    xenarch's download pipeline zero-pads the strip's final block to a full
    square (square_tiles(): 'last short block is zero-padded to square and
    flagged'), so that tile is mostly black padding, not geology — drop it
    unconditionally rather than relying on the statistical low-info gate."""
    groups: Dict[str, List[Tuple[int, Path]]] = {}
    unindexed: List[Path] = []
    for f in files:
        try:
            rel = f.relative_to(root)
        except ValueError:
            rel = Path(f.name)
        grp = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        m = re.search(r"(\d+)$", f.stem)
        if m:
            groups.setdefault(grp, []).append((int(m.group(1)), f))
        else:
            unindexed.append(f)

    drop = set()
    for grp, indexed in groups.items():
        last = max(indexed, key=lambda t: t[0])[1]
        drop.add(last)

    kept = [f for f in files if f not in drop]
    log(f"dropped {len(drop)} last-tile-per-product file(s) "
        f"(known zero-padded/black end block)")
    return kept


def prepare_chips(dataset_dir: str, chip_size: int, max_chips: int,
                  val_frac: float, test_frac: float, seed: int,
                  tmp_dir: str, log) -> Tuple[List[str], List[str], List[str]]:
    """Build a train/val/test split of .npy chips from `dataset_dir`: drop
    the last (padded) tile of every product, then apply the Mk20 low-
    information gate to what remains."""
    root = Path(dataset_dir)
    files = mk20._iter_dataset_images(root)
    if not files:
        raise SystemExit(f"no images under {root}")
    files = drop_last_tile_per_group(files, root, log)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))

    kept: List[str] = []
    n_dropped = 0
    for i in order:
        if len(kept) >= max_chips:
            break
        f = files[int(i)]
        try:
            arr = mk20._load_full_chip(f, chip_size, "tile")
        except Exception as exc:
            log(f"  skip {f.name}: {exc}")
            continue
        if arr.std() < 1e-4 or not mk20.chip_information_ok(arr)["ok"]:
            n_dropped += 1
            continue
        cp = Path(tmp_dir) / f"{f.stem}_{len(kept):04d}.npy"
        np.save(str(cp), arr.astype(np.float32))
        kept.append(str(cp))

    n_val = max(4, int(len(kept) * val_frac))
    n_test = max(4, int(len(kept) * test_frac))
    val_paths = kept[:n_val]
    test_paths = kept[n_val:n_val + n_test]
    train_paths = kept[n_val + n_test:]
    log(f"chips: {len(kept)} kept ({n_dropped} dropped, low-information) -> "
        f"{len(train_paths)} train / {len(val_paths)} val / {len(test_paths)} test")
    if len(train_paths) < 8:
        raise SystemExit(f"only {len(train_paths)} training chips after gate/split — "
                         "raise --max-chips")
    return train_paths, val_paths, test_paths


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TRAIN + EVALUATE ONE HYPERPARAMETER CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def _recon_mse(model, paths, batch_size) -> float:
    model.eval()
    tot, n = 0.0, 0
    loader = DataLoader(mk20.TorchDataset(paths, augment=False),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            mu, _ = model.encode(imgs)
            recon = model.decode(mu)
            m = torch.mean((imgs - recon) ** 2, dim=[1, 2, 3])
            tot += float(m.sum()); n += m.numel()
    model.train()
    return tot / max(n, 1)


def _recon_diagnostics(model, paths, batch_size) -> Dict[str, float]:
    """contrast / edge_regularity / gradient of what the VAE RECONSTRUCTS
    from `paths` (not of the raw chips) — the part of the Mk20 metric set
    that is actually sensitive to the hyperparameters being searched."""
    model.eval()
    scorer = mk20.NumpyAnomalyScorer()
    contrasts, edges, grads = [], [], []
    loader = DataLoader(mk20.TorchDataset(paths, augment=False),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for imgs, _ in loader:
            mu, _ = model.encode(imgs.to(DEVICE))
            recon = model.decode(mu).cpu().numpy()
            for r in recon:
                arr = r[0]
                contrasts.append(mk20.contrast_metrics(arr)["contrast"])
                edges.append(scorer.edge_regularity_score(arr))
                grads.append(scorer.gradient_score(arr))
    model.train()
    return {"contrast": float(np.mean(contrasts)),
            "edge_regularity": float(np.mean(edges)),
            "gradient": float(np.mean(grads))}


def train_eval(params: Dict, train_paths: List[str], val_paths: List[str],
              test_paths: List[str], epochs: int, chip_size: int, seed: int = 0,
              trial: Optional["optuna.Trial"] = None) -> Dict:
    """Train one VAE with `params` for up to `epochs` epochs, trimming the
    top trim_frac highest-error chips after warmup. Reports the running
    validation loss to `trial` each epoch so Optuna's successive-halving
    pruner can cut a bad trial short. Test metrics are computed once at the
    end, purely for reporting — never used for trial selection or pruning."""
    torch.manual_seed(seed)
    model = mk20.StableConvolutionalVAE(latent_dim=params["latent_dim"],
                                        input_size=chip_size).to(DEVICE)
    optim_ = torch.optim.Adam(model.parameters(), lr=params["lr"])
    warmup = max(1, epochs // 2)
    train_use = list(train_paths)
    val_loss = float("inf")
    train_curve: List[float] = []
    val_curve: List[float] = []
    t0 = time.time()
    epochs_run = 0

    for epoch in range(epochs):
        if epoch >= warmup and params["trim_frac"] > 0 and len(train_paths) > 10:
            losses = mk20.per_chip_recon_mse(model, train_paths, DEVICE, params["batch_size"])
            n_keep = max(int(len(train_paths) * (1 - params["trim_frac"])), 8)
            train_use = [train_paths[i] for i in np.argsort(losses)[:n_keep]]

        loader = DataLoader(mk20.TorchDataset(train_use, augment=True),
                            batch_size=params["batch_size"], shuffle=True, num_workers=0)
        model.train()
        kl_w = min(1.0, (epoch + 1) / warmup)
        nan_hit = False
        tot_loss, nb = 0.0, 0
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            optim_.zero_grad()
            recon, mu, logvar = model(imgs)
            loss, _, _ = mk20.stable_vae_loss(recon, imgs, mu, logvar,
                                              beta=params["beta"], kl_weight=kl_w)
            if torch.isnan(loss):
                nan_hit = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim_.step()
            tot_loss += loss.item(); nb += 1
        epochs_run = epoch + 1
        if nan_hit:
            val_loss = float("inf")
            break

        train_curve.append(tot_loss / max(nb, 1))
        val_loss = _recon_mse(model, val_paths, params["batch_size"])
        val_curve.append(val_loss)
        if trial is not None:
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    elapsed = time.time() - t0
    finite = np.isfinite(val_loss)
    diag = (_recon_diagnostics(model, val_paths, params["batch_size"]) if finite
            else {"contrast": None, "edge_regularity": None, "gradient": None})
    test_loss = _recon_mse(model, test_paths, params["batch_size"]) if finite else float("inf")
    test_diag = (_recon_diagnostics(model, test_paths, params["batch_size"]) if finite
                else {"contrast": None, "edge_regularity": None, "gradient": None})

    return {"val_loss": val_loss, "test_loss": test_loss,
            "elapsed_sec": round(elapsed, 1), "epochs_run": epochs_run,
            "train_curve": train_curve, "val_curve": val_curve,
            **diag,
            **{f"test_{k}": v for k, v in test_diag.items()}}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SEARCH SPACE + THE FOUR METHODS
# ─────────────────────────────────────────────────────────────────────────────

SPACE = dict(
    lr=(1e-4, 3e-3),          # loguniform
    latent_dim=[24, 56, 96, 128],
    batch_size=[4, 8, 16],
    trim_frac=(0.0, 0.20),    # uniform
    beta=(0.002, 0.05),       # loguniform
)


def sample_random(rng: np.random.Generator) -> Dict:
    lo, hi = SPACE["lr"]
    blo, bhi = SPACE["beta"]
    tlo, thi = SPACE["trim_frac"]
    return dict(
        lr=float(np.exp(rng.uniform(np.log(lo), np.log(hi)))),
        latent_dim=int(rng.choice(SPACE["latent_dim"])),
        batch_size=int(rng.choice(SPACE["batch_size"])),
        trim_frac=float(rng.uniform(tlo, thi)),
        beta=float(np.exp(rng.uniform(np.log(blo), np.log(bhi)))),
    )


def sample_optuna(trial: "optuna.Trial") -> Dict:
    lo, hi = SPACE["lr"]
    blo, bhi = SPACE["beta"]
    tlo, thi = SPACE["trim_frac"]
    return dict(
        lr=trial.suggest_float("lr", lo, hi, log=True),
        latent_dim=trial.suggest_categorical("latent_dim", SPACE["latent_dim"]),
        batch_size=trial.suggest_categorical("batch_size", SPACE["batch_size"]),
        trim_frac=trial.suggest_float("trim_frac", tlo, thi),
        beta=trial.suggest_float("beta", blo, bhi, log=True),
    )


def run_gradient_baseline(train_paths, val_paths, test_paths, epochs, chip_size,
                          seed, log) -> List[Dict]:
    log("GRADIENT (baseline, fixed default hyperparameters, no search)")
    r = train_eval(DEFAULT_PARAMS, train_paths, val_paths, test_paths, epochs, chip_size, seed=seed)
    r.update(method="gradient", trial=0, pruned=False, **DEFAULT_PARAMS)
    log(f"  val_loss={r['val_loss']:.5f}  test_loss={r['test_loss']:.5f}  "
        f"({r['epochs_run']} epochs, {r['elapsed_sec']}s)")
    return [r]


def run_random_search(train_paths, val_paths, test_paths, epochs, chip_size,
                      n_trials, seed, log) -> List[Dict]:
    log(f"RANDOM SEARCH ({n_trials} trials, full {epochs}-epoch budget each)")
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_trials):
        params = sample_random(rng)
        try:
            r = train_eval(params, train_paths, val_paths, test_paths, epochs, chip_size,
                           seed=seed * 1000 + i)
        except Exception as exc:
            log(f"  trial {i}: FAILED ({exc})")
            r = {"val_loss": float("inf"), "test_loss": float("inf"), "elapsed_sec": 0.0,
                "epochs_run": 0, "train_curve": [], "val_curve": [],
                "contrast": None, "edge_regularity": None, "gradient": None,
                "test_contrast": None, "test_edge_regularity": None, "test_gradient": None}
        r.update(method="random", trial=i, pruned=False, **params)
        log(f"  trial {i}: val_loss={r['val_loss']:.5f}  test_loss={r['test_loss']:.5f}  {params}")
        out.append(r)
    return out


def run_optuna_search(train_paths, val_paths, test_paths, epochs, chip_size,
                      n_trials, seed, log, pruner, method_name) -> List[Dict]:
    log(f"{method_name.upper()} ({n_trials} trials"
        f"{', successive-halving pruning' if not isinstance(pruner, NopPruner) else ''})")
    out: List[Dict] = []

    def objective(trial: "optuna.Trial") -> float:
        params = sample_optuna(trial)
        try:
            r = train_eval(params, train_paths, val_paths, test_paths, epochs, chip_size,
                           seed=seed * 1000 + trial.number, trial=trial)
            r.update(method=method_name, trial=trial.number, pruned=False, **params)
            log(f"  trial {trial.number}: val_loss={r['val_loss']:.5f}  "
                f"test_loss={r['test_loss']:.5f}  {params}")
        except optuna.TrialPruned:
            r = {"val_loss": float("inf"), "test_loss": float("inf"), "elapsed_sec": 0.0,
                "epochs_run": 0, "train_curve": [], "val_curve": [],
                "contrast": None, "edge_regularity": None, "gradient": None,
                "test_contrast": None, "test_edge_regularity": None, "test_gradient": None,
                "method": method_name, "trial": trial.number, "pruned": True, **params}
            log(f"  trial {trial.number}: PRUNED early  {params}")
        out.append(r)
        return r["val_loss"]

    study = optuna.create_study(direction="minimize",
                                sampler=TPESampler(seed=seed), pruner=pruner)
    study.optimize(objective, n_trials=n_trials)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ONE FULL SEED-RUN  (data split + all 4 methods)
# ─────────────────────────────────────────────────────────────────────────────

def run_one_seed(seed: int, args, log) -> List[Dict]:
    tmp_dir = tempfile.mkdtemp(prefix=f"finetuning20_seed{seed}_")
    train_paths, val_paths, test_paths = prepare_chips(
        args.dataset, args.chip_size, args.max_chips, args.val_frac, args.test_frac,
        seed, tmp_dir, log)

    rows: List[Dict] = []
    rows += run_gradient_baseline(train_paths, val_paths, test_paths, args.epochs,
                                  args.chip_size, seed, log)
    rows += run_random_search(train_paths, val_paths, test_paths, args.epochs,
                              args.chip_size, args.trials, seed, log)
    rows += run_optuna_search(train_paths, val_paths, test_paths, args.epochs,
                              args.chip_size, args.trials, seed, log, NopPruner(), "optuna")
    rows += run_optuna_search(train_paths, val_paths, test_paths, args.epochs,
                              args.chip_size, args.trials, seed, log,
                              SuccessiveHalvingPruner(min_resource=1, reduction_factor=2),
                              "halving")
    for r in rows:
        r["seed"] = seed
    return rows


def _best_by_val(rows: List[Dict]) -> Optional[Dict]:
    finite = [r for r in rows if np.isfinite(r["val_loss"])]
    return min(finite, key=lambda r: r["val_loss"]) if finite else None


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    v = [x for x in values if x is not None and np.isfinite(x)]
    if not v:
        return None, None
    return float(np.mean(v)), float(np.std(v))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PLOTS
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {"gradient": "#8a8a8a", "random": "#4c9be8",
         "optuna": "#e8794c", "halving": "#4ce89b"}


def plot_comparison(agg: Dict, out_path: str):
    metrics = [("test_loss", "test loss (recon MSE)"),
              ("test_contrast", "test contrast"),
              ("test_edge_regularity", "test edge_regularity"),
              ("test_gradient", "test gradient")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (key, title) in zip(axes.flat, metrics):
        means = [agg[m][key]["mean"] or 0 for m in METHODS]
        stds = [agg[m][key]["std"] or 0 for m in METHODS]
        colors = [COLORS[m] for m in METHODS]
        ax.bar(METHODS, means, yerr=stds, capsize=5, color=colors,
              edgecolor="black", linewidth=0.6)
        ax.set_title(title)
        ax.set_ylabel("mean +/- std across seeds")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Finetuning20 — method comparison (mean +/- std over seeds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss_curves(per_seed_rows: Dict[int, List[Dict]], out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, method in zip(axes.flat, METHODS):
        for seed, rows in per_seed_rows.items():
            method_rows = [r for r in rows if r["method"] == method]
            best = _best_by_val(method_rows)
            if best and best.get("val_curve"):
                xs = list(range(1, len(best["val_curve"]) + 1))
                ax.plot(xs, best["val_curve"], marker="o", label=f"seed={seed}")
        ax.set_title(method)
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation loss (recon MSE)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Finetuning20 — best-trial validation loss per epoch, all methods")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(mk20.TRAINING_DIR / "img curated dataset"))
    ap.add_argument("--trials", type=int, default=6, help="trials per searched method")
    ap.add_argument("--epochs", type=int, default=4, help="max epoch budget per trial")
    ap.add_argument("--max-chips", type=int, default=220)
    ap.add_argument("--chip-size", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seeds", default="17,53,91",
                    help="comma-separated list of distinct top-level seeds, "
                         "one full re-run of everything per seed")
    ap.add_argument("--out", default="finetuning20")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    t_start = time.time()

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    per_seed_rows: Dict[int, List[Dict]] = {}
    all_rows: List[Dict] = []
    for seed in seeds:
        log(f"===== SEED {seed} " + "=" * 50)
        rows = run_one_seed(seed, args, log)
        per_seed_rows[seed] = rows
        all_rows.extend(rows)

    # ── per-method, per-seed "best" trial (selected by validation loss) ────
    per_method_best: Dict[str, List[Dict]] = {m: [] for m in METHODS}
    for seed in seeds:
        rows = per_seed_rows[seed]
        for m in METHODS:
            best = _best_by_val([r for r in rows if r["method"] == m])
            if best is not None:
                per_method_best[m].append(best)

    agg: Dict[str, Dict] = {}
    for m in METHODS:
        bests = per_method_best[m]
        method_rows = [r for r in all_rows if r["method"] == m]
        entry = {}
        for key in ("val_loss", "test_loss", "test_contrast",
                   "test_edge_regularity", "test_gradient"):
            mean, std = _mean_std([b.get(key) for b in bests])
            entry[key] = {"mean": mean, "std": std}
        entry["n_seeds"] = len(bests)
        entry["n_trials_total"] = len(method_rows)
        entry["n_pruned_total"] = sum(1 for r in method_rows if r.get("pruned"))
        entry["total_epochs_trained"] = sum(r["epochs_run"] for r in method_rows)
        entry["total_wall_sec"] = round(sum(r["elapsed_sec"] for r in method_rows), 1)
        entry["best_params_per_seed"] = [
            {k: b[k] for k in ("lr", "latent_dim", "batch_size", "trim_frac", "beta")}
            for b in bests
        ]
        agg[m] = entry

    overall_best = min(
        (m for m in METHODS if agg[m]["test_loss"]["mean"] is not None),
        key=lambda m: agg[m]["test_loss"]["mean"], default=None)

    result = {
        "dataset": str(args.dataset), "chip_size": args.chip_size,
        "epoch_budget": args.epochs, "trials_per_method": args.trials,
        "seeds": seeds, "val_frac": args.val_frac, "test_frac": args.test_frac,
        "elapsed_sec_total": round(time.time() - t_start, 1),
        "overall_best_method": overall_best,
        "comparison": agg,
    }

    # ── write CSV (every trial, every seed) ─────────────────────────────────
    csv_path = f"{args.out}_trials.csv"
    cols = ["seed", "method", "trial", "pruned", "val_loss", "test_loss",
            "test_contrast", "test_edge_regularity", "test_gradient",
            "epochs_run", "elapsed_sec", "lr", "latent_dim", "batch_size",
            "trim_frac", "beta"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({c: r.get(c, "") for c in cols})

    json_path = f"{args.out}_comparison.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    comparison_png = f"{args.out}_comparison.png"
    plot_comparison(agg, comparison_png)
    loss_png = f"{args.out}_loss_curves.png"
    plot_loss_curves(per_seed_rows, loss_png)

    # ── console report ──────────────────────────────────────────────────────
    def fmt(entry, key):
        m, s = entry[key]["mean"], entry[key]["std"]
        return "n/a" if m is None else f"{m:.5f} +/- {s:.5f}"

    bar = "-" * 106
    print("\n" + bar)
    print(f" FINETUNING20 | seeds={seeds} | epoch budget {args.epochs} | "
         f"{args.trials} trials/method | 3-way split (train/val/test)")
    print(bar)
    print(f" {'method':<10}{'test_loss':>24}{'test_contrast':>24}"
         f"{'test_edge_reg':>24}{'test_gradient':>24}")
    for m in METHODS:
        e = agg[m]
        print(f" {m:<10}{fmt(e,'test_loss'):>24}{fmt(e,'test_contrast'):>24}"
             f"{fmt(e,'test_edge_regularity'):>24}{fmt(e,'test_gradient'):>24}")
    print(bar)
    print(f" {'method':<10}{'epochs_total':>14}{'wall_s_total':>14}{'pruned/total':>14}")
    for m in METHODS:
        e = agg[m]
        print(f" {m:<10}{e['total_epochs_trained']:>14}{e['total_wall_sec']:>14.1f}"
             f"{str(e['n_pruned_total'])+'/'+str(e['n_trials_total']):>14}")
    print(bar)
    print(f" winner (lowest mean test loss across seeds): {overall_best}")
    print(bar)
    print(f" wrote {csv_path}")
    print(f" wrote {json_path}")
    print(f" wrote {comparison_png}")
    print(f" wrote {loss_png}")
    print(bar + "\n")


if __name__ == "__main__":
    main()
