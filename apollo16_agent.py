#!/usr/bin/env python3
"""
apollo16_agent.py — targeted high-resolution imagery agent for the Apollo 16
Lunar Module "Orion" descent stage (Descartes Highlands).

This is the mirror image of lroc_fetch.py.  That harvester *avoids* landing
sites in order to build a natural-terrain reference corpus; this agent hunts
down the best available orbital imagery of one specific piece of human
hardware, so the Xenarch autoencoder has a known-truth positive control to be
scored against.

What "agent" means here — a loop with a fallback at every stage that keeps a
journal of what it decided and why:

    PLAN    resolve the target point and the search box around it
    SEARCH  find frames covering the point
              1. ODE REST (oderest.rsl.wustl.edu) — rich geometry metadata
              2. PDS cumulative index (pds.lroc.asu.edu) — authoritative fallback
    PROBE   HTTP Range-read each candidate's attached PDS3 label (~64 KB)
            rather than pulling a 250 MB frame just to read its header
    RANK    score candidates on ground sample distance + illumination geometry
    LOCATE  estimate the target's (line, sample) inside each ranked frame
    FETCH   Range-download only the line span around the target
    VERIFY  heuristic check that the cut-out holds a bright compact object with
            an attached shadow — a sanity check, NOT a detection claim
    REPORT  PNG + .npy chips, manifest.json for xenarch_pipeline, JSON journal

Usage:
  python apollo16_agent.py                      # full run, best 3 frames
  python apollo16_agent.py --dry-run            # search + rank + report only
  python apollo16_agent.py --top 5 --window-px 4096
  python apollo16_agent.py --source pds         # skip ODE, index only
  python apollo16_agent.py --product-type CDRNAC
  python apollo16_agent.py --self-test          # offline checks, no network

Output:
  <output>/frames/       Range-fetched line spans (.npy) + PNG previews
  <output>/chips/        chips centred on the target, .npy + .png
  <output>/manifest.json chip list in lroc_fetch.py's schema
  <output>/journal.json  every candidate, score and decision

Georeferencing caveat: the locator interpolates the frame corner coordinates
published in the PDS index (or, failing that, projects from the frame centre
using NORTH_AZIMUTH).  Both are approximations good to roughly a hundred
metres, which is why the default crop window is ~1 km across.  For metre-level
registration run the ISIS3 commands printed by --emit-isis.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("apollo16_agent")


# ── the target ─────────────────────────────────────────────────────────────

@dataclass
class Target:
    name: str
    lat: float          # degrees, +N
    lon: float          # degrees east, 0-360
    note: str = ""


# LROC NAC-derived position of the Apollo 16 LM descent stage at Descartes.
# Good to a few tens of metres, which is irrelevant at our crop width.
APOLLO16_LM = Target(
    name="Apollo 16 LM descent stage (Orion)",
    lat=-8.9730,
    lon=15.5002,
    note="Descartes Highlands. Descent stage body ~4.2 m across, landing gear "
         "spread ~9.4 m footpad to footpad.",
)

# Radius that still contains the ALSEP, the LRV's final parking spot and the
# near-LM traverse — everything worth having in the same cut-out.
SITE_CONTEXT_RADIUS_M = 600.0

MOON_RADIUS_M = 1_737_400.0

# NAC instantaneous field of view: 0.5 m/pixel from the 50 km nominal orbit.
NAC_IFOV_RAD = 1.0e-5
NAC_SAMPLES = 5064


# ── ranking policy ─────────────────────────────────────────────────────────

# Weighted the same way as the pipeline's combined_weights blocks, and
# overridable from a JSON file via --weights.
RANK_WEIGHTS: Dict[str, float] = {
    "resolution": 0.55,   # metres per pixel — the whole point of the exercise
    "incidence":  0.25,   # sun elevation; shadows are what make hardware visible
    "emission":   0.20,   # off-nadir smears the target and wrecks the geometry
}

RES_FLOOR_M   = 0.20   # about the best NAC has ever managed over an Apollo site
RES_CEILING_M = 2.00   # past this the 9.4 m gear spread is under 5 px
INCIDENCE_IDEAL = (40.0, 75.0)   # degrees; long shadows, target still lit
INCIDENCE_HARD  = (10.0, 85.0)   # outside this, score 0
EMISSION_SOFT   = 10.0           # degrees; no penalty below this
EMISSION_HARD   = 40.0           # above this, score 0


# ── endpoints ──────────────────────────────────────────────────────────────

ODE_BASE  = "https://oderest.rsl.wustl.edu/live2/"
PDS_BASE  = "https://pds.lroc.asu.edu/data/LRO-L-LROC-2-EDR-V1.0"
PDS_ROOT  = "https://pds.lroc.asu.edu/data"

LABEL_PROBE_BYTES = 65536
HTTP_TIMEOUT      = 60
INDEX_TIMEOUT     = 300


# ── candidates ─────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    product_id: str
    url: str
    source: str                       # "ode" | "pds_index"
    lat: Optional[float] = None       # frame centre
    lon: Optional[float] = None
    resolution_m: Optional[float] = None
    incidence_deg: Optional[float] = None
    emission_deg: Optional[float] = None
    phase_deg: Optional[float] = None
    north_azimuth_deg: Optional[float] = None
    altitude_km: Optional[float] = None
    utc: Optional[str] = None
    lines: Optional[int] = None
    samples: Optional[int] = None
    corners: Optional[Dict[str, float]] = None   # ul/ur/ll/lr lat+lon
    score: float = 0.0
    score_parts: Dict[str, float] = field(default_factory=dict)
    est_line: Optional[float] = None
    est_sample: Optional[float] = None
    locate_method: str = ""
    notes: List[str] = field(default_factory=list)


# ── small helpers ──────────────────────────────────────────────────────────

def norm_lon(lon: float) -> float:
    """Longitude into [0, 360)."""
    return lon % 360.0


def lon_delta(lon: float, lon0: float) -> float:
    """Signed east-west separation in degrees, wrap-safe, in (-180, 180]."""
    return ((lon - lon0 + 180.0) % 360.0) - 180.0


def metres_to_deg_lat(m: float) -> float:
    return math.degrees(m / MOON_RADIUS_M)


def metres_to_deg_lon(m: float, lat: float) -> float:
    c = max(math.cos(math.radians(lat)), 1e-6)
    return math.degrees(m / (MOON_RADIUS_M * c))


def _to_float(v) -> Optional[float]:
    try:
        f = float(str(v).strip().split()[0])
    except (TypeError, ValueError, IndexError):
        return None
    return None if math.isnan(f) else f


def _first(d: Dict, *names: str):
    """
    Case/underscore-insensitive lookup across several possible key spellings.
    ODE has renamed fields between releases and PDS labels differ per product
    type, so every metadata read goes through here.
    """
    if not isinstance(d, dict):
        return None
    flat = {re.sub(r"[^a-z0-9]", "", k.lower()): v for k, v in d.items()}
    for n in names:
        key = re.sub(r"[^a-z0-9]", "", n.lower())
        if key in flat and flat[key] not in ("", None, "N/A", "NULL"):
            return flat[key]
    return None


# ── PDS3 label parsing ─────────────────────────────────────────────────────

_KV_RE = re.compile(r"^\s*(\^?[A-Za-z0-9_:]+)\s*=\s*(.+?)\s*$", re.MULTILINE)


def parse_pds_label(text: str) -> Dict[str, str]:
    """
    Flatten a PDS3 label into {KEYWORD: raw value}.  Nested OBJECT/GROUP
    structure is dropped — every keyword we need is uniquely named within a
    NAC label, and a flat dict survives the layout differences between EDR,
    CDR and the index labels.  Later duplicates win, which matches the
    convention that the IMAGE object trails the file-level keywords.
    """
    out: Dict[str, str] = {}
    body = text.split("\nEND\r\n")[0].split("\nEND\n")[0]
    for m in _KV_RE.finditer(body):
        key, val = m.group(1).upper(), m.group(2).strip()
        if key in ("OBJECT", "END_OBJECT", "GROUP", "END_GROUP"):
            continue
        out[key] = val.strip('"')
    return out


def label_float(label: Dict[str, str], *names: str) -> Optional[float]:
    for n in names:
        if n in label:
            f = _to_float(re.sub(r"<[^>]*>", "", label[n]))
            if f is not None:
                return f
    return None


def label_corners(label: Dict[str, str]) -> Optional[Dict[str, float]]:
    """Pull the four corner lat/lons if the label or index row carries them."""
    keys = {
        "ul_lat": ("UPPER_LEFT_LATITUDE",),
        "ul_lon": ("UPPER_LEFT_LONGITUDE",),
        "ur_lat": ("UPPER_RIGHT_LATITUDE",),
        "ur_lon": ("UPPER_RIGHT_LONGITUDE",),
        "ll_lat": ("LOWER_LEFT_LATITUDE",),
        "ll_lon": ("LOWER_LEFT_LONGITUDE",),
        "lr_lat": ("LOWER_RIGHT_LATITUDE",),
        "lr_lon": ("LOWER_RIGHT_LONGITUDE",),
    }
    out: Dict[str, float] = {}
    for k, names in keys.items():
        v = label_float(label, *names)
        if v is None:
            return None
        out[k] = v
    return out


# ── raw image layout ───────────────────────────────────────────────────────

@dataclass
class ImageLayout:
    lines: int
    samples: int
    dtype: str            # numpy dtype string, e.g. ">u1"
    bytes_per_sample: int
    data_offset: int      # byte offset of the first image line
    row_stride: int       # bytes from one line to the next
    prefix_bytes: int
    range_safe: bool      # False when the label hints at compression
    note: str = ""


_SAMPLE_TYPES = {
    "LSB_INTEGER":            "<i",
    "MSB_INTEGER":            ">i",
    "PC_INTEGER":             "<i",
    "SUN_INTEGER":            ">i",
    "UNSIGNED_INTEGER":       ">u",
    "MSB_UNSIGNED_INTEGER":   ">u",
    "LSB_UNSIGNED_INTEGER":   "<u",
    "PC_UNSIGNED_INTEGER":    "<u",
    "PC_REAL":                "<f",
    "IEEE_REAL":              ">f",
    "SUN_REAL":               ">f",
}


def image_layout(label: Dict[str, str]) -> Optional[ImageLayout]:
    """
    Work out where the pixels live inside a fixed-record PDS3 file, so that a
    single line span can be Range-requested instead of the whole frame.
    """
    lines   = label_float(label, "LINES", "IMAGE_LINES")
    samples = label_float(label, "LINE_SAMPLES", "SAMPLES")
    bits    = label_float(label, "SAMPLE_BITS") or 8.0
    if not lines or not samples:
        return None

    stype = str(label.get("SAMPLE_TYPE", "MSB_UNSIGNED_INTEGER")).upper()
    base  = _SAMPLE_TYPES.get(stype, ">u")
    nbytes = max(int(bits) // 8, 1)
    dtype  = f"{base}{nbytes}"

    prefix = int(label_float(label, "LINE_PREFIX_BYTES") or 0)
    suffix = int(label_float(label, "LINE_SUFFIX_BYTES") or 0)
    rec    = int(label_float(label, "RECORD_BYTES") or 0)

    row = prefix + int(samples) * nbytes + suffix
    if str(label.get("RECORD_TYPE", "")).upper().startswith("FIXED") and rec >= row:
        row = rec

    # ^IMAGE is either a record number (1-based) or "(file, record)".
    offset = 0
    ptr = label.get("^IMAGE", "")
    m = re.search(r"(\d+)", str(ptr))
    if m and rec:
        offset = (int(m.group(1)) - 1) * rec
    elif "<BYTES>" in str(ptr).upper() and m:
        offset = int(m.group(1)) - 1

    encoding = str(label.get("ENCODING_TYPE", "N/A")).upper()
    range_safe = encoding in ("", "N/A", "NONE")

    return ImageLayout(
        lines=int(lines), samples=int(samples), dtype=dtype,
        bytes_per_sample=nbytes, data_offset=offset, row_stride=row,
        prefix_bytes=prefix, range_safe=range_safe,
        note="" if range_safe else f"ENCODING_TYPE={encoding}; partial reads unsafe",
    )


def decode_line_span(raw: bytes, layout: ImageLayout, n_lines: int) -> np.ndarray:
    """Turn a Range-fetched byte span into a (n_lines, samples) float array."""
    usable = n_lines * layout.row_stride
    if len(raw) < usable:
        n_lines = len(raw) // layout.row_stride
        usable = n_lines * layout.row_stride
    if n_lines <= 0:
        return np.zeros((0, layout.samples), dtype=np.float32)

    rows = np.frombuffer(raw[:usable], dtype=np.uint8).reshape(n_lines, layout.row_stride)
    px_bytes = layout.samples * layout.bytes_per_sample
    body = rows[:, layout.prefix_bytes : layout.prefix_bytes + px_bytes]
    arr = np.frombuffer(body.tobytes(), dtype=np.dtype(layout.dtype))
    return arr.reshape(n_lines, layout.samples).astype(np.float32)


# ── geolocation ────────────────────────────────────────────────────────────

def bilinear_forward(corners: Dict[str, float], u: float, v: float,
                     lon0: float) -> Tuple[float, float]:
    """
    Map normalised frame coordinates to (dlon, lat).  u runs across samples
    (0 = left), v down lines (0 = top).  Longitudes are carried as signed
    offsets from lon0 so the frame may straddle the prime meridian.
    """
    dul = lon_delta(corners["ul_lon"], lon0)
    dur = lon_delta(corners["ur_lon"], lon0)
    dll = lon_delta(corners["ll_lon"], lon0)
    dlr = lon_delta(corners["lr_lon"], lon0)
    top_lon = dul + (dur - dul) * u
    bot_lon = dll + (dlr - dll) * u
    top_lat = corners["ul_lat"] + (corners["ur_lat"] - corners["ul_lat"]) * u
    bot_lat = corners["ll_lat"] + (corners["lr_lat"] - corners["ll_lat"]) * u
    return (top_lon + (bot_lon - top_lon) * v,
            top_lat + (bot_lat - top_lat) * v)


def bilinear_inverse(corners: Dict[str, float], lat: float, lon: float,
                     iterations: int = 40) -> Optional[Tuple[float, float]]:
    """
    Invert the corner bilinear map by Newton iteration: given a lat/lon,
    return (u, v) in normalised frame coordinates.  Returns None if it fails
    to converge — the caller then falls back to the tangent-plane estimate.
    """
    lon0 = corners["ul_lon"]
    target_dlon = lon_delta(lon, lon0)
    u = v = 0.5
    eps = 1e-6
    for _ in range(iterations):
        f_lon, f_lat = bilinear_forward(corners, u, v, lon0)
        r_lon, r_lat = f_lon - target_dlon, f_lat - lat
        if abs(r_lon) < 1e-9 and abs(r_lat) < 1e-9:
            break
        du_lon, du_lat = bilinear_forward(corners, u + eps, v, lon0)
        dv_lon, dv_lat = bilinear_forward(corners, u, v + eps, lon0)
        j11, j21 = (du_lon - f_lon) / eps, (du_lat - f_lat) / eps
        j12, j22 = (dv_lon - f_lon) / eps, (dv_lat - f_lat) / eps
        det = j11 * j22 - j12 * j21
        if abs(det) < 1e-15:
            return None
        u -= ( j22 * r_lon - j12 * r_lat) / det
        v -= (-j21 * r_lon + j11 * r_lat) / det
        u = min(max(u, -1.0), 2.0)
        v = min(max(v, -1.0), 2.0)
    f_lon, f_lat = bilinear_forward(corners, u, v, lon0)
    if abs(f_lon - target_dlon) > 1e-4 or abs(f_lat - lat) > 1e-4:
        return None
    return u, v


def tangent_plane_locate(cand: Candidate, target: Target) -> Optional[Tuple[float, float]]:
    """
    Fallback when no corner coordinates exist: project the target onto a plane
    tangent at the frame centre, then rotate by NORTH_AZIMUTH to get pixels.
    NORTH_AZIMUTH is the bearing of lunar north in the image measured clockwise
    from the +sample (right) direction, so north sits at that angle and the
    line axis increases downward.
    """
    if cand.lat is None or cand.lon is None or not cand.resolution_m:
        return None
    if not cand.lines or not cand.samples:
        return None

    dn = math.radians(target.lat - cand.lat) * MOON_RADIUS_M
    de = (math.radians(lon_delta(target.lon, cand.lon)) * MOON_RADIUS_M
          * math.cos(math.radians(cand.lat)))

    na = math.radians(cand.north_azimuth_deg if cand.north_azimuth_deg is not None else 90.0)
    # Unit vectors, in (sample, line) pixel space, of north and of east.
    n_s, n_l = math.cos(na), -math.sin(na)
    e_s, e_l = math.cos(na - math.pi / 2), -math.sin(na - math.pi / 2)

    ds = (dn * n_s + de * e_s) / cand.resolution_m
    dl = (dn * n_l + de * e_l) / cand.resolution_m
    return cand.samples / 2.0 + ds, cand.lines / 2.0 + dl


def locate_target(cand: Candidate, target: Target) -> Tuple[Optional[float], Optional[float], str]:
    """Best available (line, sample) estimate, plus which method produced it."""
    if cand.corners:
        uv = bilinear_inverse(cand.corners, target.lat, target.lon)
        if uv is not None and cand.lines and cand.samples:
            u, v = uv
            return v * cand.lines, u * cand.samples, "corner_bilinear"
    tp = tangent_plane_locate(cand, target)
    if tp is not None:
        s, l = tp
        return l, s, "tangent_plane"
    return None, None, "none"


# ── ranking ────────────────────────────────────────────────────────────────

def _resolution_term(res: Optional[float]) -> float:
    if res is None:
        return 0.35            # unknown: mid-low, so metadata-rich frames win
    if res <= RES_FLOOR_M:
        return 1.0
    if res >= RES_CEILING_M:
        return 0.0
    return (RES_CEILING_M - res) / (RES_CEILING_M - RES_FLOOR_M)


def _incidence_term(inc: Optional[float]) -> float:
    if inc is None:
        return 0.4
    lo_hard, hi_hard = INCIDENCE_HARD
    lo_ideal, hi_ideal = INCIDENCE_IDEAL
    if inc <= lo_hard or inc >= hi_hard:
        return 0.0
    if lo_ideal <= inc <= hi_ideal:
        return 1.0
    if inc < lo_ideal:
        return (inc - lo_hard) / (lo_ideal - lo_hard)
    return (hi_hard - inc) / (hi_hard - hi_ideal)


def _emission_term(emi: Optional[float]) -> float:
    if emi is None:
        return 0.4
    if emi <= EMISSION_SOFT:
        return 1.0
    if emi >= EMISSION_HARD:
        return 0.0
    return (EMISSION_HARD - emi) / (EMISSION_HARD - EMISSION_SOFT)


def rank_candidates(cands: Sequence[Candidate],
                    weights: Optional[Dict[str, float]] = None) -> List[Candidate]:
    w = dict(RANK_WEIGHTS)
    if weights:
        w.update(weights)
    total = sum(w.values()) or 1.0
    for c in cands:
        parts = {
            "resolution": _resolution_term(c.resolution_m),
            "incidence":  _incidence_term(c.incidence_deg),
            "emission":   _emission_term(c.emission_deg),
        }
        c.score_parts = parts
        c.score = sum(parts[k] * w.get(k, 0.0) for k in parts) / total
    return sorted(cands, key=lambda c: (-c.score, c.resolution_m or 9e9, c.product_id))


# ── verification heuristic ─────────────────────────────────────────────────

def _box_mean(a: np.ndarray, k: int = 3) -> np.ndarray:
    """k×k mean via sliding windows; numpy only, no scipy dependency."""
    if a.shape[0] < k or a.shape[1] < k:
        return a.copy()
    win = np.lib.stride_tricks.sliding_window_view(a, (k, k))
    return win.mean(axis=(-1, -2))


def score_lander_signature(chip: np.ndarray) -> Dict[str, float]:
    """
    Heuristic: does this cut-out contain a small, very bright object with a
    dark patch beside it?  That is what a metal descent stage plus its shadow
    looks like at half-metre resolution.

    This is a triage aid for ordering human review, not a detection — natural
    boulders produce the same signature, which is precisely why the Xenarch
    autoencoder exists.
    """
    a = np.asarray(chip, dtype=np.float32)
    if a.size == 0 or a.shape[0] < 8 or a.shape[1] < 8:
        return {"bright_sigma": 0.0, "shadow_sigma": 0.0,
                "compactness": 0.0, "heuristic": 0.0}

    sm = _box_mean(a, 3)
    med = float(np.median(sm))
    mad = float(np.median(np.abs(sm - med))) * 1.4826
    if mad < 1e-8:
        return {"bright_sigma": 0.0, "shadow_sigma": 0.0,
                "compactness": 0.0, "heuristic": 0.0}

    py, px = np.unravel_index(int(np.argmax(sm)), sm.shape)
    peak = float(sm[py, px])
    bright_sigma = (peak - med) / mad

    # Darkest point in a ring 2-14 px out from the peak: the attached shadow.
    r = 14
    y0, y1 = max(py - r, 0), min(py + r + 1, sm.shape[0])
    x0, x1 = max(px - r, 0), min(px + r + 1, sm.shape[1])
    patch = sm[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist = np.sqrt((yy - py) ** 2 + (xx - px) ** 2)
    ring = patch[(dist >= 2) & (dist <= r)]
    shadow_sigma = (med - float(ring.min())) / mad if ring.size else 0.0

    # Compact: the bright pixels should be few and clustered near the peak.
    thresh = med + 0.5 * (peak - med)
    hot = sm >= thresh
    n_hot = int(hot.sum())
    near = int(hot[y0:y1, x0:x1].sum())
    compactness = (near / n_hot) if n_hot else 0.0

    heuristic = (
        min(bright_sigma / 8.0, 1.0) * 0.45 +
        min(max(shadow_sigma, 0.0) / 4.0, 1.0) * 0.35 +
        compactness * 0.20
    )
    return {
        "bright_sigma": round(bright_sigma, 3),
        "shadow_sigma": round(shadow_sigma, 3),
        "compactness":  round(compactness, 3),
        "heuristic":    round(float(min(max(heuristic, 0.0), 1.0)), 3),
        "peak_line":    float(py),
        "peak_sample":  float(px),
    }


# ── search backends ────────────────────────────────────────────────────────

class ODESearch:
    """
    PDS Geosciences ODE REST.  One spatial query returns every LROC product
    overlapping the box, with the illumination geometry already computed —
    far cheaper than crawling the archive.
    """

    def __init__(self, session: requests.Session, product_types: Sequence[str]):
        self.session = session
        self.product_types = product_types

    def search(self, target: Target, half_width_deg: float,
               limit: int = 200) -> List[Candidate]:
        out: List[Candidate] = []
        for pt in self.product_types:
            params = {
                "query": "product",
                "target": "moon",
                "ihid": "LRO",
                "iid": "LROC",
                "pt": pt,
                "westernlon": round(norm_lon(target.lon - half_width_deg), 5),
                "easternlon": round(norm_lon(target.lon + half_width_deg), 5),
                "minlatitude": round(target.lat - half_width_deg, 5),
                "maxlatitude": round(target.lat + half_width_deg, 5),
                "results": "fmp",
                "output": "JSON",
                "limit": limit,
            }
            try:
                r = self.session.get(ODE_BASE, params=params, timeout=HTTP_TIMEOUT)
                r.raise_for_status()
                payload = r.json()
            except Exception as e:
                log.warning(f"  ODE query failed for {pt}: {e}")
                continue

            products = self._products(payload)
            log.info(f"  ODE {pt}: {len(products)} products")
            for p in products:
                c = self._to_candidate(p, pt)
                if c:
                    out.append(c)
        return out

    @staticmethod
    def _products(payload: Dict) -> List[Dict]:
        resp = payload.get("ODEResults", payload)
        prods = _first(resp, "Products") or {}
        items = _first(prods, "Product") if isinstance(prods, dict) else prods
        if items is None:
            return []
        return items if isinstance(items, list) else [items]

    @staticmethod
    def _img_url(p: Dict) -> str:
        """Prefer the .IMG in Product_files; fall back to the product URL."""
        files = _first(p, "Product_files") or {}
        entries = _first(files, "Product_file") if isinstance(files, dict) else files
        if entries is not None:
            if not isinstance(entries, list):
                entries = [entries]
            for f in entries:
                url = _first(f, "URL", "FileURL", "PDSFileName") or ""
                if str(url).upper().endswith(".IMG"):
                    return str(url)
        return str(_first(p, "ProductURL", "LabelURL", "PDSVolumeURL") or "")

    def _to_candidate(self, p: Dict, pt: str) -> Optional[Candidate]:
        pid = str(_first(p, "pdsid", "ProductId", "PDSID") or "").strip()
        url = self._img_url(p)
        if not pid or not url:
            return None
        corners = {}
        for key, names in (
            ("ul_lat", ("UL_Latitude", "Corner1Latitude")),
            ("ul_lon", ("UL_Longitude", "Corner1Longitude")),
            ("ur_lat", ("UR_Latitude", "Corner2Latitude")),
            ("ur_lon", ("UR_Longitude", "Corner2Longitude")),
            ("lr_lat", ("LR_Latitude", "Corner3Latitude")),
            ("lr_lon", ("LR_Longitude", "Corner3Longitude")),
            ("ll_lat", ("LL_Latitude", "Corner4Latitude")),
            ("ll_lon", ("LL_Longitude", "Corner4Longitude")),
        ):
            v = _to_float(_first(p, *names))
            if v is None:
                corners = {}
                break
            corners[key] = v

        return Candidate(
            product_id=pid,
            url=url,
            source=f"ode:{pt}",
            lat=_to_float(_first(p, "Center_latitude", "CenterLatitude")),
            lon=(lambda v: norm_lon(v) if v is not None else None)(
                _to_float(_first(p, "Center_longitude", "CenterLongitude"))),
            resolution_m=_to_float(_first(p, "Map_resolution", "Pixel_resolution",
                                          "Image_resolution", "Resolution")),
            incidence_deg=_to_float(_first(p, "Incidence_angle", "IncidenceAngle")),
            emission_deg=_to_float(_first(p, "Emission_angle", "EmissionAngle")),
            phase_deg=_to_float(_first(p, "Phase_angle", "PhaseAngle")),
            utc=str(_first(p, "UTC_start_time", "Observation_time") or "") or None,
            lines=(lambda v: int(v) if v else None)(
                _to_float(_first(p, "Lines", "Image_lines"))),
            samples=(lambda v: int(v) if v else None)(
                _to_float(_first(p, "Samples", "Line_samples"))),
            corners=corners or None,
        )


class PDSIndexSearch:
    """
    Fallback: stream the LROC cumulative index and keep the rows whose frame
    footprint contains the target.  Column offsets are read from CUMINDEX.LBL
    rather than hardcoded, so this keeps working when the archive's column
    layout changes.  CUMINDEX is cumulative through its volume, so scanning
    the newest volume first usually settles the search in one download.
    """

    def __init__(self, session: requests.Session, cache_dir: Path,
                 max_vols: int = 2):
        self.session = session
        self.cache_dir = cache_dir
        self.max_vols = max_vols
        cache_dir.mkdir(parents=True, exist_ok=True)

    # -- label-driven column map -------------------------------------------
    @staticmethod
    def parse_index_label(text: str) -> Tuple[Dict[str, Tuple[int, int]], int]:
        """
        Return {COLUMN_NAME: (start, stop)} as 0-based Python slice bounds,
        plus RECORD_BYTES.  PDS START_BYTE is 1-based.
        """
        rec = 0
        m = re.search(r"^\s*RECORD_BYTES\s*=\s*(\d+)", text, re.MULTILINE)
        if m:
            rec = int(m.group(1))

        cols: Dict[str, Tuple[int, int]] = {}
        for block in re.split(r"OBJECT\s*=\s*COLUMN", text)[1:]:
            nm = re.search(r"^\s*NAME\s*=\s*\"?([A-Za-z0-9_:]+)", block, re.MULTILINE)
            sb = re.search(r"^\s*START_BYTE\s*=\s*(\d+)", block, re.MULTILINE)
            by = re.search(r"^\s*BYTES\s*=\s*(\d+)", block, re.MULTILINE)
            if nm and sb and by:
                start = int(sb.group(1)) - 1
                cols[nm.group(1).upper()] = (start, start + int(by.group(1)))
        return cols, rec

    @staticmethod
    def _cell(row: bytes, cols: Dict[str, Tuple[int, int]], *names: str) -> Optional[str]:
        for n in names:
            span = cols.get(n.upper())
            if not span:
                continue
            try:
                return row[span[0]:span[1]].decode("ascii", "ignore").strip().strip('"')
            except Exception:
                return None
        return None

    def _volumes(self) -> List[str]:
        r = self.session.get(PDS_BASE + "/", timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        vols = sorted({m.group(0).rstrip("/")
                       for m in re.finditer(r"LROLRC_\d+/", r.text)}, reverse=True)
        return vols[: self.max_vols]

    def _scan(self, vol: str, target: Target, half_width_deg: float) -> List[Candidate]:
        cache = self.cache_dir / f"{vol}_apollo16.json"
        if cache.exists():
            log.info(f"  {vol}: using cached index matches")
            return [Candidate(**c) for c in json.loads(cache.read_text())]

        lbl_url = f"{PDS_BASE}/{vol}/INDEX/CUMINDEX.LBL"
        tab_url = f"{PDS_BASE}/{vol}/INDEX/CUMINDEX.TAB"
        try:
            lbl = self.session.get(lbl_url, timeout=HTTP_TIMEOUT)
            lbl.raise_for_status()
            cols, rec = self.parse_index_label(lbl.text)
        except Exception as e:
            log.warning(f"  {vol}: cannot read CUMINDEX.LBL — {e}")
            return []
        if not cols or not rec:
            log.warning(f"  {vol}: unusable index label")
            return []

        log.info(f"  {vol}: streaming index ({len(cols)} columns, {rec} B/row) …")
        found: List[Candidate] = []
        try:
            with self.session.get(tab_url, stream=True, timeout=INDEX_TIMEOUT) as r:
                r.raise_for_status()
                buf = b""
                for chunk in r.iter_content(chunk_size=1 << 20):
                    buf += chunk
                    while len(buf) >= rec:
                        row, buf = buf[:rec], buf[rec:]
                        c = self._row_to_candidate(row, cols, target, half_width_deg)
                        if c:
                            found.append(c)
        except Exception as e:
            log.warning(f"  {vol}: index stream failed — {e}")

        cache.write_text(json.dumps([asdict(c) for c in found], indent=2))
        log.info(f"  {vol}: {len(found)} frames over the target")
        return found

    def _row_to_candidate(self, row: bytes, cols: Dict[str, Tuple[int, int]],
                          target: Target, half: float) -> Optional[Candidate]:
        fspec = self._cell(row, cols, "FILE_SPECIFICATION_NAME")
        if not fspec or "/NAC/" not in fspec.upper():
            return None
        lat = _to_float(self._cell(row, cols, "CENTER_LATITUDE"))
        lon = _to_float(self._cell(row, cols, "CENTER_LONGITUDE"))
        if lat is None or lon is None:
            return None
        lon = norm_lon(lon)

        corners = {}
        for key, col in (("ul_lat", "UPPER_LEFT_LATITUDE"),
                         ("ul_lon", "UPPER_LEFT_LONGITUDE"),
                         ("ur_lat", "UPPER_RIGHT_LATITUDE"),
                         ("ur_lon", "UPPER_RIGHT_LONGITUDE"),
                         ("ll_lat", "LOWER_LEFT_LATITUDE"),
                         ("ll_lon", "LOWER_LEFT_LONGITUDE"),
                         ("lr_lat", "LOWER_RIGHT_LATITUDE"),
                         ("lr_lon", "LOWER_RIGHT_LONGITUDE")):
            v = _to_float(self._cell(row, cols, col))
            if v is None:
                corners = {}
                break
            corners[key] = v

        if corners:
            lats = [corners[k] for k in corners if k.endswith("lat")]
            dlons = [lon_delta(corners[k], target.lon) for k in corners if k.endswith("lon")]
            inside = (min(lats) <= target.lat <= max(lats)
                      and min(dlons) <= 0.0 <= max(dlons))
        else:
            inside = (abs(lat - target.lat) <= half
                      and abs(lon_delta(lon, target.lon)) <= half)
        if not inside:
            return None

        alt = _to_float(self._cell(row, cols, "SPACECRAFT_ALTITUDE"))
        emi = _to_float(self._cell(row, cols, "EMISSION_ANGLE"))
        res = _to_float(self._cell(row, cols, "RESOLUTION", "PIXEL_SCALE"))
        if res is None and alt:
            # NAC IFOV is 10 µrad; slant range stretches the footprint.
            res = alt * 1000.0 * NAC_IFOV_RAD
            if emi is not None and 0 <= emi < 80:
                res /= max(math.cos(math.radians(emi)), 0.2)

        pid = Path(fspec).stem
        return Candidate(
            product_id=pid,
            url=f"{PDS_ROOT}/{fspec.lstrip('/')}",
            source="pds_index",
            lat=lat, lon=lon,
            resolution_m=res,
            incidence_deg=_to_float(self._cell(row, cols, "INCIDENCE_ANGLE")),
            emission_deg=emi,
            phase_deg=_to_float(self._cell(row, cols, "PHASE_ANGLE")),
            north_azimuth_deg=_to_float(self._cell(row, cols, "NORTH_AZIMUTH")),
            altitude_km=alt,
            utc=self._cell(row, cols, "START_TIME"),
            lines=(lambda v: int(v) if v else None)(
                _to_float(self._cell(row, cols, "IMAGE_LINES", "LINES"))),
            samples=(lambda v: int(v) if v else None)(
                _to_float(self._cell(row, cols, "LINE_SAMPLES", "SAMPLES"))),
            corners=corners or None,
        )

    def search(self, target: Target, half_width_deg: float) -> List[Candidate]:
        try:
            vols = self._volumes()
        except Exception as e:
            log.error(f"  Cannot list PDS volumes: {e}")
            return []
        log.info(f"  Scanning {len(vols)} volume(s), newest first: {', '.join(vols)}")
        out: List[Candidate] = []
        seen = set()
        for v in vols:
            for c in self._scan(v, target, half_width_deg):
                if c.product_id not in seen:
                    seen.add(c.product_id)
                    out.append(c)
            if out:
                break   # CUMINDEX is cumulative; one hit-bearing volume is enough
        return out


# ── the agent ──────────────────────────────────────────────────────────────

class Apollo16Agent:

    def __init__(self, target: Target, out_dir: Path, session: requests.Session,
                 top: int = 3, window_px: int = 2048, chip_size: int = 256,
                 sources: str = "both", product_types: Sequence[str] = ("EDRNAC",),
                 half_width_deg: float = 0.15, dry_run: bool = False,
                 weights: Optional[Dict[str, float]] = None,
                 max_vols: int = 2,
                 refine_line: Optional[float] = None,
                 refine_sample: Optional[float] = None):
        self.target = target
        self.out = out_dir
        self.session = session
        self.top = top
        self.window_px = window_px
        self.chip_size = chip_size
        self.sources = sources
        self.product_types = product_types
        self.half_width_deg = half_width_deg
        self.dry_run = dry_run
        self.weights = weights
        self.max_vols = max_vols
        # Pixel coordinates from an ISIS campt run, overriding the estimator.
        self.refine_line = refine_line
        self.refine_sample = refine_sample
        self.journal: List[Dict] = []
        self.chips: List[Dict] = []

    def note(self, phase: str, **kw):
        entry = {"phase": phase, "t": time.strftime("%H:%M:%S"), **kw}
        self.journal.append(entry)

    # -- PLAN --------------------------------------------------------------
    def plan(self):
        half_m = SITE_CONTEXT_RADIUS_M
        log.info("═" * 66)
        log.info(f"TARGET  {self.target.name}")
        log.info(f"        {abs(self.target.lat):.4f}°{'S' if self.target.lat < 0 else 'N'}  "
                 f"{norm_lon(self.target.lon):.4f}°E")
        log.info(f"        {self.target.note}")
        log.info(f"        search box ±{self.half_width_deg}°  "
                 f"(~{self.half_width_deg * math.radians(1) * MOON_RADIUS_M / 1000:.0f} km)")
        log.info(f"        site context radius {half_m:.0f} m")
        log.info("═" * 66)
        self.note("plan", target=asdict(self.target),
                  half_width_deg=self.half_width_deg,
                  window_px=self.window_px, chip_size=self.chip_size)

    # -- SEARCH ------------------------------------------------------------
    def search(self) -> List[Candidate]:
        cands: List[Candidate] = []
        if self.sources in ("ode", "both"):
            log.info("SEARCH  ODE REST …")
            try:
                cands += ODESearch(self.session, self.product_types).search(
                    self.target, self.half_width_deg)
            except Exception as e:
                log.warning(f"  ODE backend unavailable: {e}")
        if not cands and self.sources in ("pds", "both"):
            log.info("SEARCH  PDS cumulative index …")
            try:
                cands += PDSIndexSearch(
                    self.session, self.out / "index_cache", self.max_vols
                ).search(self.target, self.half_width_deg)
            except Exception as e:
                log.warning(f"  PDS backend unavailable: {e}")

        dedup: Dict[str, Candidate] = {}
        for c in cands:
            dedup.setdefault(c.product_id, c)
        out = list(dedup.values())
        log.info(f"SEARCH  {len(out)} unique candidate frames")
        self.note("search", n_candidates=len(out),
                  product_ids=[c.product_id for c in out][:50])
        return out

    # -- PROBE -------------------------------------------------------------
    def probe(self, cand: Candidate) -> Optional[Tuple[Dict[str, str], ImageLayout]]:
        """Range-read the attached PDS3 label instead of the whole frame."""
        try:
            r = self.session.get(
                cand.url, timeout=HTTP_TIMEOUT,
                headers={"Range": f"bytes=0-{LABEL_PROBE_BYTES - 1}"})
            if r.status_code not in (200, 206):
                r.raise_for_status()
            head = r.content[:LABEL_PROBE_BYTES]
            cand.notes.append("range_supported" if r.status_code == 206
                              else "no_range_support")
        except Exception as e:
            log.warning(f"  {cand.product_id}: label probe failed — {e}")
            cand.notes.append(f"probe_failed: {e}")
            return None

        label = parse_pds_label(head.decode("latin-1", "ignore"))
        layout = image_layout(label)

        cand.lines   = cand.lines   or (layout.lines if layout else None)
        cand.samples = cand.samples or (layout.samples if layout else None)
        cand.corners = cand.corners or label_corners(label)
        cand.incidence_deg = cand.incidence_deg if cand.incidence_deg is not None \
            else label_float(label, "INCIDENCE_ANGLE")
        cand.emission_deg = cand.emission_deg if cand.emission_deg is not None \
            else label_float(label, "EMISSION_ANGLE")
        cand.north_azimuth_deg = cand.north_azimuth_deg if cand.north_azimuth_deg is not None \
            else label_float(label, "NORTH_AZIMUTH")
        cand.altitude_km = cand.altitude_km if cand.altitude_km is not None \
            else label_float(label, "SPACECRAFT_ALTITUDE")
        if cand.resolution_m is None:
            r_m = label_float(label, "RESOLUTION", "PIXEL_SCALE")
            if r_m is None and cand.altitude_km:
                r_m = cand.altitude_km * 1000.0 * NAC_IFOV_RAD
            cand.resolution_m = r_m
        if layout and not layout.range_safe:
            cand.notes.append(layout.note)
        return (label, layout) if layout else None

    # -- FETCH -------------------------------------------------------------
    def fetch_span(self, cand: Candidate, layout: ImageLayout,
                   l0: int, l1: int) -> Optional[np.ndarray]:
        n = l1 - l0
        start = layout.data_offset + l0 * layout.row_stride
        end   = start + n * layout.row_stride - 1
        mb    = n * layout.row_stride / 1e6
        log.info(f"  Fetching lines {l0}-{l1} of {cand.product_id} "
                 f"({mb:.1f} MB of {layout.lines * layout.row_stride / 1e6:.0f} MB)")
        try:
            r = self.session.get(cand.url, timeout=HTTP_TIMEOUT,
                                 headers={"Range": f"bytes={start}-{end}"},
                                 stream=True)
            if r.status_code == 206:
                return decode_line_span(r.content, layout, n)
            log.warning(f"  {cand.product_id}: server ignored Range "
                        f"(HTTP {r.status_code}); falling back to full read")
            r.close()
        except Exception as e:
            log.warning(f"  {cand.product_id}: range fetch failed — {e}")
            return None

        try:
            with self.session.get(cand.url, timeout=INDEX_TIMEOUT, stream=True) as r:
                r.raise_for_status()
                blob = r.content
            usable = blob[start : start + n * layout.row_stride]
            return decode_line_span(usable, layout, n)
        except Exception as e:
            log.error(f"  {cand.product_id}: full download failed — {e}")
            return None

    # -- crop / save -------------------------------------------------------
    @staticmethod
    def _stretch(a: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(a, [1, 99])
        return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)

    def _save_png(self, arr01: np.ndarray, path: Path):
        if not HAS_PIL:
            return
        Image.fromarray((arr01 * 255).astype(np.uint8)).save(path)

    def harvest(self, cand: Candidate) -> Optional[Dict]:
        probed = self.probe(cand)
        if not probed:
            return None
        _, layout = probed

        if self.refine_line is not None and self.refine_sample is not None:
            line, sample, method = self.refine_line, self.refine_sample, "manual_refine"
        else:
            line, sample, method = locate_target(cand, self.target)
        cand.est_line, cand.est_sample, cand.locate_method = line, sample, method
        if line is None:
            log.warning(f"  {cand.product_id}: cannot locate the target — skipping")
            return None
        log.info(f"  {cand.product_id}: target near line {line:.0f}, "
                 f"sample {sample:.0f}  ({method})")

        if not (0 <= line < layout.lines and 0 <= sample < layout.samples):
            log.warning(f"  {cand.product_id}: estimate lands outside the frame "
                        f"({layout.lines}×{layout.samples}) — skipping")
            cand.notes.append("estimate_outside_frame")
            return None

        half = self.window_px // 2
        l0 = max(int(line) - half, 0)
        l1 = min(int(line) + half, layout.lines)
        if self.dry_run:
            log.info(f"  [dry-run] would fetch lines {l0}-{l1}")
            return None

        span = self.fetch_span(cand, layout, l0, l1)
        if span is None or span.size == 0:
            return None

        frames_dir = self.out / "frames"
        chips_dir  = self.out / "chips"
        frames_dir.mkdir(parents=True, exist_ok=True)
        chips_dir.mkdir(parents=True, exist_ok=True)

        # Window: the full sample width is only 5064 px, so keep it all —
        # cross-track context costs nothing and absorbs locator error.
        win = self._stretch(span)
        np.save(frames_dir / f"{cand.product_id}_window.npy", win)
        self._save_png(win, frames_dir / f"{cand.product_id}_window.png")

        # Chip: tight cut-out centred on the estimate, for the pipeline.
        cl = int(line) - l0
        cs = int(sample)
        h = self.chip_size // 2
        y0, y1 = max(cl - h, 0), min(cl + h, win.shape[0])
        x0, x1 = max(cs - h, 0), min(cs + h, win.shape[1])
        chip = win[y0:y1, x0:x1]
        chip_path = chips_dir / f"{cand.product_id}_apollo16_lm.npy"
        np.save(chip_path, chip)
        self._save_png(chip, chips_dir / f"{cand.product_id}_apollo16_lm.png")

        sig = score_lander_signature(chip)
        log.info(f"  {cand.product_id}: signature heuristic {sig['heuristic']:.2f} "
                 f"(bright {sig['bright_sigma']:.1f}σ, shadow {sig['shadow_sigma']:.1f}σ)")

        gear_px = (9.4 / cand.resolution_m) if cand.resolution_m else None
        record = {
            "chip_path":     str(chip_path),
            "source":        f"{cand.product_id}.IMG",
            "chip_id":       0,
            "center_x":      cs,
            "center_y":      int(line),
            "target":        self.target.name,
            "target_lat":    self.target.lat,
            "target_lon":    norm_lon(self.target.lon),
            "resolution_m":  cand.resolution_m,
            "gear_span_px":  round(gear_px, 1) if gear_px else None,
            "incidence_deg": cand.incidence_deg,
            "emission_deg":  cand.emission_deg,
            "locate_method": cand.locate_method,
            "rank_score":    round(cand.score, 4),
            "signature":     sig,
            "window_png":    str(frames_dir / f"{cand.product_id}_window.png"),
            "url":           cand.url,
        }
        self.chips.append(record)
        return record

    # -- REPORT ------------------------------------------------------------
    def write_outputs(self, ranked: List[Candidate]):
        self.out.mkdir(parents=True, exist_ok=True)

        manifest = {
            "created":     time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_chips": len(self.chips),
            "chip_size":   self.chip_size,
            "description": ("Apollo 16 LM descent stage — positive control for the "
                            "Xenarch anomaly detector. These chips contain known "
                            "artificial structure and should score as anomalies."),
            "target":      asdict(self.target),
            "chips":       self.chips,
        }
        (self.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

        journal = {
            "created":    time.strftime("%Y-%m-%dT%H:%M:%S"),
            "target":     asdict(self.target),
            "weights":    self.weights or RANK_WEIGHTS,
            "candidates": [asdict(c) for c in ranked],
            "journal":    self.journal,
        }
        (self.out / "journal.json").write_text(json.dumps(journal, indent=2))
        log.info(f"Wrote {self.out}/manifest.json and {self.out}/journal.json")

    def emit_isis(self, ranked: List[Candidate]):
        """Print the ISIS3 recipe for metre-level registration of the best frame."""
        if not ranked:
            return
        c = ranked[0]
        print("\n# Refine the pixel location with ISIS3 (needs ISIS + LRO SPICE kernels):")
        print(f"lronac2isis from={c.product_id}.IMG to={c.product_id}.cub")
        print(f"spiceinit from={c.product_id}.cub")
        print(f"lronaccal from={c.product_id}.cub to={c.product_id}.cal.cub")
        print(f"campt from={c.product_id}.cal.cub type=ground "
              f"latitude={self.target.lat} longitude={norm_lon(self.target.lon)}")
        print("# campt reports the exact Line/Sample; re-run with "
              "--refine-line/--refine-sample to cut the chip there.\n")

    # -- main loop ---------------------------------------------------------
    def run(self) -> List[Dict]:
        self.plan()
        cands = self.search()
        if not cands:
            log.error("No candidate frames found. Try --source pds, widen "
                      "--half-width, or check network access to the archives.")
            self.note("search", result="empty")
            self.write_outputs([])
            return []

        ranked = rank_candidates(cands, self.weights)
        log.info("RANK    best candidates:")
        for i, c in enumerate(ranked[: max(self.top * 3, 10)], 1):
            log.info(f"  {i:2d}. {c.product_id:<16} score={c.score:.3f}  "
                     f"res={c.resolution_m if c.resolution_m else float('nan'):.2f} m/px  "
                     f"inc={c.incidence_deg if c.incidence_deg is not None else float('nan'):.1f}°  "
                     f"emi={c.emission_deg if c.emission_deg is not None else float('nan'):.1f}°  "
                     f"[{c.source}]")
        self.note("rank", ranked=[{"id": c.product_id, "score": round(c.score, 4),
                                   "res_m": c.resolution_m} for c in ranked[:20]])

        taken = 0
        for c in ranked:
            if taken >= self.top:
                break
            log.info(f"FETCH   {c.product_id} …")
            rec = self.harvest(c)
            if rec:
                taken += 1
        self.note("fetch", chips=len(self.chips))

        self.write_outputs(ranked)
        if self.chips:
            best = max(self.chips, key=lambda r: r["signature"]["heuristic"])
            log.info("─" * 66)
            log.info(f"Best cut-out: {best['window_png']}")
            log.info(f"  {best['resolution_m']:.2f} m/px → the 9.4 m landing gear "
                     f"spread spans ~{best['gear_span_px']:.0f} px")
            log.info(f"  signature heuristic {best['signature']['heuristic']:.2f} "
                     "(triage only — confirm by eye)")
            log.info("─" * 66)
        elif not self.dry_run:
            log.warning("No chips produced. The candidates were found but could not "
                        "be read; see journal.json for per-frame notes.")
        return self.chips


# ── offline self-test ──────────────────────────────────────────────────────

def self_test() -> int:
    """Exercise the parsing, geometry and scoring logic without any network."""
    failures: List[str] = []

    def check(name: str, ok: bool, detail: str = ""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("apollo16_agent self-test")

    # 1. index label → column map
    lbl = """
RECORD_BYTES = 901
OBJECT = COLUMN
  NAME = FILE_SPECIFICATION_NAME
  START_BYTE = 16
  BYTES = 75
END_OBJECT = COLUMN
OBJECT = COLUMN
  NAME = CENTER_LATITUDE
  START_BYTE = 806
  BYTES = 6
END_OBJECT = COLUMN
"""
    cols, rec = PDSIndexSearch.parse_index_label(lbl)
    check("index label columns", cols.get("CENTER_LATITUDE") == (805, 811) and rec == 901,
          f"{cols}, rec={rec}")

    # 2. attached PDS3 label → layout
    pds = ('PDS_VERSION_ID = PDS3\r\nRECORD_TYPE = FIXED_LENGTH\r\n'
           'RECORD_BYTES = 5080\r\n^IMAGE = 3\r\nOBJECT = IMAGE\r\n'
           '  LINES = 52224\r\n  LINE_SAMPLES = 5064\r\n  SAMPLE_BITS = 8\r\n'
           '  SAMPLE_TYPE = MSB_UNSIGNED_INTEGER\r\n  LINE_PREFIX_BYTES = 16\r\n'
           'END_OBJECT = IMAGE\r\nEND\r\n')
    label = parse_pds_label(pds)
    lay = image_layout(label)
    check("pds label → layout",
          lay is not None and lay.samples == 5064 and lay.row_stride == 5080
          and lay.data_offset == 2 * 5080 and lay.prefix_bytes == 16 and lay.range_safe,
          f"stride={lay.row_stride if lay else None} offset={lay.data_offset if lay else None}")

    # 3. raw line span decode, prefix bytes stripped
    if lay:
        small = ImageLayout(lines=4, samples=6, dtype=">u1", bytes_per_sample=1,
                            data_offset=0, row_stride=10, prefix_bytes=4,
                            range_safe=True)
        raw = b"".join(bytes([9, 9, 9, 9]) + bytes([i] * 6) for i in range(4))
        dec = decode_line_span(raw, small, 4)
        check("line span decode", dec.shape == (4, 6) and dec[2, 0] == 2 and dec.max() == 3,
              f"shape={dec.shape}")

    # 4. corner bilinear round trip
    corners = {"ul_lat": -8.80, "ul_lon": 15.40, "ur_lat": -8.81, "ur_lon": 15.62,
               "ll_lat": -9.15, "ll_lon": 15.38, "lr_lat": -9.16, "lr_lon": 15.60}
    u0, v0 = 0.37, 0.62
    dlon, lat = bilinear_forward(corners, u0, v0, corners["ul_lon"])
    uv = bilinear_inverse(corners, lat, corners["ul_lon"] + dlon)
    check("bilinear inverse round trip",
          uv is not None and abs(uv[0] - u0) < 1e-4 and abs(uv[1] - v0) < 1e-4,
          f"{uv}")

    # 5. locate_target puts the Apollo 16 point inside a frame that covers it
    c = Candidate(product_id="TEST", url="", source="test",
                  lat=-8.98, lon=15.50, lines=52224, samples=5064, corners=corners)
    line, sample, method = locate_target(c, APOLLO16_LM)
    check("locate via corners",
          method == "corner_bilinear" and line is not None
          and 0 < line < 52224 and 0 < sample < 5064,
          f"line={line:.0f} sample={sample:.0f}" if line else "no fix")

    # 6. tangent-plane fallback agrees with the corner fix to within ~2 km
    c2 = Candidate(product_id="TEST2", url="", source="test", lat=-8.98, lon=15.50,
                   lines=52224, samples=5064, resolution_m=0.5, north_azimuth_deg=90.0)
    l2, s2, m2 = locate_target(c2, APOLLO16_LM)
    err_m = math.hypot((l2 - 52224 / 2) * 0.5, (s2 - 5064 / 2) * 0.5) if l2 else 9e9
    check("tangent-plane fallback", m2 == "tangent_plane" and err_m < 2000,
          f"{err_m:.0f} m from frame centre")

    # 7. ranking prefers the sharper, better-lit frame
    a = Candidate("SHARP", "", "t", resolution_m=0.28, incidence_deg=55, emission_deg=3)
    b = Candidate("COARSE", "", "t", resolution_m=1.6, incidence_deg=20, emission_deg=35)
    ranked = rank_candidates([b, a])
    check("ranking order", ranked[0].product_id == "SHARP",
          f"{[(r.product_id, round(r.score, 3)) for r in ranked]}")

    # 8. signature heuristic: bright blob + shadow beats flat noise
    rng = np.random.default_rng(16)
    flat = rng.normal(0.5, 0.02, (128, 128)).astype(np.float32)
    lander = flat.copy()
    lander[62:66, 62:66] = 1.0          # sunlit descent stage
    lander[62:66, 68:80] = 0.05         # its shadow
    s_flat = score_lander_signature(flat)["heuristic"]
    s_land = score_lander_signature(lander)["heuristic"]
    check("signature heuristic", s_land > s_flat + 0.2,
          f"lander={s_land:.2f} flat={s_flat:.2f}")

    # 9. longitude wrap safety
    check("longitude wrap", abs(lon_delta(1.0, 359.0) - 2.0) < 1e-9
          and abs(norm_lon(-8.5) - 351.5) < 1e-9)

    # 10. ODE response shape → candidate (the live service is not exercised here)
    payload = {"ODEResults": {"Products": {"Product": [{
        "pdsid": "M175179080LE",
        "Center_latitude": "-8.9711", "Center_longitude": "15.5033",
        "Map_resolution": "0.31", "Incidence_angle": "56.2",
        "Emission_angle": "1.4", "Phase_angle": "57.1",
        "UTC_start_time": "2011-11-08T12:00:00",
        "UL_Latitude": "-8.80", "UL_Longitude": "15.40",
        "UR_Latitude": "-8.81", "UR_Longitude": "15.62",
        "LR_Latitude": "-9.16", "LR_Longitude": "15.60",
        "LL_Latitude": "-9.15", "LL_Longitude": "15.38",
        "Product_files": {"Product_file": [
            {"URL": "https://example.invalid/M175179080LE.LBL"},
            {"URL": "https://example.invalid/M175179080LE.IMG"}]}}]}}}
    ode = ODESearch(None, ["EDRNAC"])
    prods = ode._products(payload["ODEResults"])
    cand = ode._to_candidate(prods[0], "EDRNAC") if prods else None
    check("ODE payload → candidate",
          cand is not None and cand.url.endswith(".IMG")
          and cand.resolution_m == 0.31 and cand.corners is not None
          and abs(cand.lon - 15.5033) < 1e-6,
          f"{cand.product_id if cand else None} {cand.url if cand else ''}")

    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 0 if not failures else 1


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Find and cut out the best available orbital imagery of the "
                    "Apollo 16 LM descent stage.")
    ap.add_argument("--output", default="apollo16_imagery",
                    help="Output directory (default ./apollo16_imagery)")
    ap.add_argument("--source", choices=["ode", "pds", "both"], default="both",
                    help="Catalogue backend (default both: ODE first, PDS index as fallback)")
    ap.add_argument("--product-type", default="EDRNAC",
                    help="ODE product type(s), comma separated (EDRNAC, CDRNAC)")
    ap.add_argument("--top", type=int, default=3, help="Frames to cut out (default 3)")
    ap.add_argument("--window-px", type=int, default=2048,
                    help="Line span to fetch around the target (default 2048 ≈ 1 km)")
    ap.add_argument("--chip-size", type=int, default=256,
                    help="Chip edge in pixels for the pipeline (default 256)")
    ap.add_argument("--half-width", type=float, default=0.15,
                    help="Search box half-width in degrees (default 0.15 ≈ 4.5 km)")
    ap.add_argument("--max-vols", type=int, default=2,
                    help="PDS volumes to scan; CUMINDEX is cumulative (default 2)")
    ap.add_argument("--lat", type=float, default=None, help="Override target latitude")
    ap.add_argument("--lon", type=float, default=None, help="Override target longitude (°E)")
    ap.add_argument("--refine-line", type=float, default=None,
                    help="Use this image line instead of the estimate (from ISIS campt)")
    ap.add_argument("--refine-sample", type=float, default=None,
                    help="Use this image sample instead of the estimate (from ISIS campt)")
    ap.add_argument("--weights", default=None,
                    help="JSON file overriding the ranking weights")
    ap.add_argument("--dry-run", action="store_true",
                    help="Search, rank and report without downloading pixels")
    ap.add_argument("--emit-isis", action="store_true",
                    help="Print the ISIS3 recipe for metre-level registration")
    ap.add_argument("--self-test", action="store_true",
                    help="Run offline logic checks and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if (args.refine_line is None) != (args.refine_sample is None):
        ap.error("--refine-line and --refine-sample must be given together")
    if args.refine_line is not None and args.top != 1:
        log.warning("--refine-line/--refine-sample are frame-specific; "
                    "forcing --top 1 so they apply to the best-ranked frame only")
        args.top = 1

    target = Target(
        name=APOLLO16_LM.name,
        lat=args.lat if args.lat is not None else APOLLO16_LM.lat,
        lon=norm_lon(args.lon) if args.lon is not None else norm_lon(APOLLO16_LM.lon),
        note=APOLLO16_LM.note,
    )

    weights = None
    if args.weights:
        weights = json.loads(Path(args.weights).read_text())
        weights = weights.get("rank_weights", weights)

    session = requests.Session()
    session.headers["User-Agent"] = "XenarchApollo16Agent/1.0 (planetary research)"

    agent = Apollo16Agent(
        target=target,
        out_dir=Path(args.output),
        session=session,
        top=args.top,
        window_px=args.window_px,
        chip_size=args.chip_size,
        sources=args.source,
        product_types=[p.strip() for p in args.product_type.split(",") if p.strip()],
        half_width_deg=args.half_width,
        dry_run=args.dry_run,
        weights=weights,
        max_vols=args.max_vols,
        refine_line=args.refine_line,
        refine_sample=args.refine_sample,
    )
    chips = agent.run()

    if args.emit_isis:
        ranked = [Candidate(**c) for c in
                  json.loads((Path(args.output) / "journal.json").read_text())["candidates"]]
        agent.emit_isis(ranked)

    if chips:
        print(f"\nFeed the positive control to xenarch_pipeline with:")
        print(f"  manifest = json.load(open('{args.output}/manifest.json'))")
        print(f"  test_paths = [c['chip_path'] for c in manifest['chips']]")
    return 0 if (chips or args.dry_run) else 1


if __name__ == "__main__":
    sys.exit(main())
