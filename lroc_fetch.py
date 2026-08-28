#!/usr/bin/env python3
"""
lroc_fetch.py — Reference terrain harvester for the Xenarch pipeline.

Downloads natural-terrain chips from:
  1. NASA LRO NAC  — direct PDS HTTP archive (no auth required)
  2. ISRO Ch-2 OHRC — PRADAN portal (set PRADAN_USER / PRADAN_PASS env vars)

Usage:
  python lroc_fetch.py                          # both sources, 5 frames each
  python lroc_fetch.py --source lroc            # LRO NAC only
  python lroc_fetch.py --source ohrc            # OHRC only
  python lroc_fetch.py --frames 10              # 10 frames per source
  python lroc_fetch.py --output /path/corpus    # custom output dir
  python lroc_fetch.py --dry-run                # list candidates, no download

Output:
  <output>/lroc_nac/     downloaded .IMG files + extracted chip .npy files
  <output>/ch2_ohrc/     downloaded files + chips
  <output>/manifest.json chip paths + metadata for xenarch_pipeline reference_paths
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from tqdm import tqdm

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import pvl
    HAS_PVL = True
except ImportError:
    HAS_PVL = False

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lroc_fetch")

# ── constants ──────────────────────────────────────────────────────────────

PDS_BASE     = "https://pds.lroc.asu.edu/data/LRO-L-LROC-2-EDR-V1.0"
PDS_ROOT     = "https://pds.lroc.asu.edu/data"   # fspec paths are relative to here
PRADAN_BASE = "https://pradan.issdc.gov.in/pradan"

CHIP_SIZE       = 256
MAX_CHIPS_PER_IMAGE = 200
DOWNLOAD_TIMEOUT    = 120   # seconds per file
LABEL_TIMEOUT       = 15

# Landing sites to avoid — (lon_0_360, lat, name)
LANDING_SITES = [
    (23.5,   0.7,  "Apollo 11"),
    (336.6, -3.0,  "Apollo 12"),
    (342.5, -3.6,  "Apollo 14"),
    (3.6,   26.1,  "Apollo 15"),
    (15.5,  -9.0,  "Apollo 16"),
    (30.8,  20.2,  "Apollo 17"),
    (340.5,  2.5,  "Surveyor 3"),
    (177.6, -45.4, "Chang'e 4"),
    (308.1,  43.1, "Chang'e 5"),
    (32.4,  -69.4, "Vikram (Ch-3)"),
]
EXCLUSION_RADIUS_DEG = 2.0   # degrees around each landing site

# Reference bounding boxes (west_lon°, east_lon°, min_lat°, max_lat°) — 0-360
# Chosen for terrain variety, well away from any known landing site.
REFERENCE_REGIONS: List[Tuple[float, float, float, float, str]] = [
    (315.0, 325.0,  38.0,  48.0, "Mare Imbrium N border"),
    (170.0, 180.0, -30.0, -20.0, "Far-side S highlands"),
    (150.0, 160.0,  -5.0,   5.0, "Far-side equatorial"),
]


# ── HTML directory parser ──────────────────────────────────────────────────

class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v and not v.startswith("?") and v != "/data/":
                    self.links.append(v)


def list_directory(url: str, session: requests.Session) -> List[str]:
    """Return hrefs listed in an Apache-style directory page."""
    r = session.get(url.rstrip("/") + "/", timeout=LABEL_TIMEOUT)
    r.raise_for_status()
    p = _LinkParser()
    p.feed(r.text)
    return p.links


# ── PVL / label parsing ────────────────────────────────────────────────────

def parse_label(label_text: str) -> Dict:
    """Extract key fields from a PDS3 label string."""
    fields: Dict = {}
    for key in ("CENTER_LATITUDE", "CENTER_LONGITUDE",
                "MAXIMUM_LATITUDE", "MINIMUM_LATITUDE",
                "EASTERNMOST_LONGITUDE", "WESTERNMOST_LONGITUDE",
                "LINE_SAMPLES", "LINES", "SAMPLE_BITS"):
        m = re.search(rf"^\s*{key}\s*=\s*([^\r\n<]+)", label_text, re.MULTILINE)
        if m:
            val = m.group(1).strip().rstrip("<degrees>").strip()
            try:
                fields[key] = float(val)
            except ValueError:
                fields[key] = val
    return fields


# ── geometry helpers ───────────────────────────────────────────────────────

def in_reference_region(lon: float, lat: float) -> bool:
    """True if (lon, lat) falls inside any reference bounding box."""
    lon = lon % 360
    for w, e, s, n, _ in REFERENCE_REGIONS:
        if w <= lon <= e and s <= lat <= n:
            return True
    return False


def near_landing_site(lon: float, lat: float) -> bool:
    lon = lon % 360
    for slon, slat, name in LANDING_SITES:
        if abs(lon - slon) < EXCLUSION_RADIUS_DEG and abs(lat - slat) < EXCLUSION_RADIUS_DEG:
            return True
    return False


# ── image loading ──────────────────────────────────────────────────────────

def load_image(path: str) -> Optional[np.ndarray]:
    """
    Load a PDS3 IMG (or PNG/JPG) to a float32 [0,1] grayscale array.
    Tries rasterio first (handles PDS3 + GeoTIFF), falls back to Pillow.
    """
    if HAS_RASTERIO:
        try:
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
            lo, hi = np.percentile(arr, [1, 99])
            return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
        except Exception as e:
            log.debug(f"rasterio failed for {path}: {e}")

    try:
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.float32) / 255.0
    except Exception as e:
        log.warning(f"Pillow also failed for {path}: {e}")
        return None


# ── chip extraction ────────────────────────────────────────────────────────

def extract_chips(image_path: str, chip_dir: Path,
                  chip_size: int = CHIP_SIZE,
                  max_chips: int = MAX_CHIPS_PER_IMAGE) -> List[Dict]:
    arr = load_image(image_path)
    if arr is None:
        return []

    h, w = arr.shape
    if h < chip_size or w < chip_size:
        log.warning(f"  Image {Path(image_path).name} too small ({h}×{w}), skipping")
        return []

    stem  = Path(image_path).stem
    chips = []
    cid   = 0
    for y in range(0, h - chip_size + 1, chip_size):
        for x in range(0, w - chip_size + 1, chip_size):
            if cid >= max_chips:
                break
            chip = arr[y : y + chip_size, x : x + chip_size]
            if chip.std() < 0.01:
                continue
            fpath = chip_dir / f"{stem}_chip_{cid:04d}.npy"
            np.save(str(fpath), chip)
            chips.append({
                "chip_path": str(fpath),
                "source":    Path(image_path).name,
                "chip_id":   cid,
                "center_x":  x + chip_size // 2,
                "center_y":  y + chip_size // 2,
            })
            cid += 1
        if cid >= max_chips:
            break

    log.info(f"  Extracted {len(chips)} chips from {Path(image_path).name}")
    return chips


# ── streaming download ─────────────────────────────────────────────────────

def download_file(url: str, dest: Path, session: requests.Session,
                  dry_run: bool = False) -> bool:
    if dest.exists():
        log.info(f"  Already have {dest.name}, skipping download")
        return True
    if dry_run:
        log.info(f"  [dry-run] Would download {url}")
        return False
    log.info(f"  Downloading {dest.name} …")
    try:
        with session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                desc=dest.name, leave=False
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as e:
        log.error(f"  Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


# ── LRO NAC harvester ─────────────────────────────────────────────────────

# CUMINDEX.TAB fixed-width column layout (0-indexed byte offsets within row)
_COL_FILE  = (15, 90)   # FILE_SPECIFICATION_NAME  start=16 bytes=75 (1-indexed)
_COL_INST  = (99, 103)  # INSTRUMENT_ID            start=100 bytes=4
_COL_LAT   = (805, 811) # CENTER_LATITUDE          start=806 bytes=6
_COL_LON   = (812, 818) # CENTER_LONGITUDE         start=813 bytes=6
_ROW_BYTES = 901        # RECORD_BYTES (includes line terminator)


class NACHarvester:
    """
    Streams CUMINDEX.TAB from each PDS volume to find NAC frames covering
    the reference regions — no directory crawling required.
    """

    def __init__(self, out_dir: Path, session: requests.Session,
                 frames: int = 5, dry_run: bool = False, max_vols: int = 10):
        self.out_dir  = out_dir
        self.session  = session
        self.target   = frames
        self.dry_run  = dry_run
        self.max_vols = max_vols
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "chips").mkdir(exist_ok=True)

    def _scan_index(self, vol: str) -> List[Dict]:
        """
        Download (and cache) one volume's CUMINDEX.TAB then return matching rows.
        Cache lives in out_dir/../index_cache/ so repeated runs skip the download.
        """
        cache_dir = self.out_dir.parent / "index_cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{vol}_matches.json"

        if cache_file.exists():
            log.info(f"  {vol}: using cached index ({cache_file.name})")
            return json.loads(cache_file.read_text())

        url = f"{PDS_BASE}/{vol}/INDEX/CUMINDEX.TAB"
        matches: List[Dict] = []
        log.info(f"  {vol}: downloading index (this takes ~30-60 s, cached after) …")
        try:
            with self.session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                buf   = b""
                with tqdm(total=total, unit="B", unit_scale=True,
                          desc=f"{vol} index", leave=False) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 17):
                        buf += chunk
                        bar.update(len(chunk))
                        while len(buf) >= _ROW_BYTES:
                            row = buf[:_ROW_BYTES]
                            buf = buf[_ROW_BYTES:]
                            try:
                                fspec = row[_COL_FILE[0]:_COL_FILE[1]].decode().strip().strip('"')
                                # NAC files live under a NAC/ subdirectory; WAC under WAC/
                                if "/NAC/" not in fspec:
                                    continue
                                lat = float(row[_COL_LAT[0]:_COL_LAT[1]].decode().strip())
                                lon = float(row[_COL_LON[0]:_COL_LON[1]].decode().strip()) % 360
                            except (ValueError, UnicodeDecodeError):
                                continue
                            if not in_reference_region(lon, lat):
                                continue
                            if near_landing_site(lon, lat):
                                continue
                            matches.append({"vol": vol, "fspec": fspec,
                                            "lat": lat, "lon": lon})
        except Exception as e:
            log.warning(f"  Index scan failed for {vol}: {e}")
            return []

        cache_file.write_text(json.dumps(matches, indent=2))
        log.info(f"  {vol}: {len(matches)} matches cached → {cache_file.name}")
        return matches

    def run(self) -> List[Dict]:
        all_chips: List[Dict] = []
        downloaded = 0

        log.info("NACHarvester: listing PDS volumes …")
        try:
            volumes = sorted(
                v.rstrip("/") for v in list_directory(PDS_BASE, self.session)
                if v.startswith("LROLRC_")
            )[:self.max_vols]
        except Exception as e:
            log.error(f"Cannot reach PDS archive: {e}")
            return []
        log.info(f"Scanning {len(volumes)} volumes (indexes cached after first run) …")

        for vol in volumes:
            if downloaded >= self.target:
                break
            log.info(f"  Scanning {vol} …")
            candidates = self._scan_index(vol)
            if not candidates:
                continue
            log.info(f"  {len(candidates)} candidates in {vol}")

            for cand in candidates:
                if downloaded >= self.target:
                    break
                # fspec looks like: DATA/XXXX/MXXXXXXXXXX.IMG
                # fspec already contains the full path from PDS_ROOT
                # e.g. "LRO-L-LROC-2-EDR-V1.0/LROLRC_0001/DATA/COM/2009181/NAC/M101013931LE.IMG"
                img_url  = f"{PDS_ROOT}/{cand['fspec']}"
                img_name = Path(cand["fspec"]).name
                log.info(f"  Candidate: {img_name}  lat={cand['lat']:.1f}  lon={cand['lon']:.1f}")

                dest = self.out_dir / img_name
                ok   = download_file(img_url, dest, self.session, self.dry_run)
                if not ok:
                    continue

                chips = extract_chips(str(dest), self.out_dir / "chips")
                all_chips.extend(chips)
                downloaded += 1
                log.info(f"  [{downloaded}/{self.target}] done")

        log.info(f"NACHarvester: {downloaded} frames, {len(all_chips)} chips total")
        return all_chips


# ── Chandrayaan-2 OHRC harvester ───────────────────────────────────────────

class OHRCHarvester:
    """
    Downloads Ch-2 OHRC frames from ISRO PRADAN.
    Requires PRADAN_USER and PRADAN_PASS environment variables.
    Registration: https://pradan.issdc.gov.in/pradan/  (free)
    """

    LOGIN_URL  = f"{PRADAN_BASE}/login"
    SEARCH_URL = f"{PRADAN_BASE}/ch2/ohrc/search"

    def __init__(self, out_dir: Path, session: requests.Session,
                 frames: int = 5, dry_run: bool = False):
        self.out_dir  = out_dir
        self.session  = session
        self.target   = frames
        self.dry_run  = dry_run
        self.user     = os.environ.get("PRADAN_USER", "")
        self.password = os.environ.get("PRADAN_PASS", "")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "chips").mkdir(exist_ok=True)

    def _login(self) -> bool:
        if not self.user or not self.password:
            log.warning(
                "OHRC: PRADAN_USER / PRADAN_PASS not set — skipping OHRC.\n"
                "  Register free at https://pradan.issdc.gov.in/pradan/\n"
                "  then re-run with those env vars exported."
            )
            return False
        try:
            r = self.session.post(
                self.LOGIN_URL,
                data={"username": self.user, "password": self.password},
                timeout=20,
                allow_redirects=True,
            )
            if "logout" in r.text.lower() or r.status_code == 200:
                log.info("OHRC: PRADAN login succeeded")
                return True
            log.error(f"OHRC: login failed (HTTP {r.status_code})")
            return False
        except Exception as e:
            log.error(f"OHRC: login error — {e}")
            return False

    def _search(self, region: Tuple) -> List[Dict]:
        w, e, s, n, label = region
        # Convert back to ±180 for PRADAN
        w = w if w <= 180 else w - 360
        e = e if e <= 180 else e - 360
        try:
            r = self.session.get(
                self.SEARCH_URL,
                params={
                    "westlon": w, "eastlon": e,
                    "minlat":  s, "maxlat":  n,
                    "format":  "json",
                },
                timeout=20,
            )
            r.raise_for_status()
            return r.json().get("products", [])
        except Exception as e:
            log.warning(f"OHRC search failed for {label}: {e}")
            return []

    def run(self) -> List[Dict]:
        if not self._login():
            return []

        all_chips: List[Dict] = []
        downloaded = 0

        for region in REFERENCE_REGIONS:
            if downloaded >= self.target:
                break
            products = self._search(region)
            log.info(f"OHRC: {len(products)} products in {region[4]}")

            for prod in products:
                if downloaded >= self.target:
                    break
                url  = prod.get("download_url") or prod.get("url", "")
                name = prod.get("product_id", f"ohrc_{downloaded:04d}") + ".IMG"
                if not url:
                    continue

                lat = float(prod.get("center_lat", 0))
                lon = float(prod.get("center_lon", 0)) % 360
                if near_landing_site(lon, lat):
                    continue

                dest = self.out_dir / name
                ok   = download_file(url, dest, self.session, self.dry_run)
                if not ok:
                    continue

                chips = extract_chips(str(dest), self.out_dir / "chips")
                all_chips.extend(chips)
                downloaded += 1
                log.info(f"  [{downloaded}/{self.target}] done")

        log.info(f"OHRCHarvester: {downloaded} frames, {len(all_chips)} chips total")
        return all_chips


# ── manifest ───────────────────────────────────────────────────────────────

def write_manifest(out_dir: Path, all_chips: List[Dict]) -> Path:
    manifest = {
        "created":     time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_chips": len(all_chips),
        "chip_size":   CHIP_SIZE,
        "description": "Natural lunar terrain reference corpus — no artificial structures",
        "chips":       all_chips,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    log.info(f"Manifest written → {path}  ({len(all_chips)} chips)")
    return path


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Xenarch reference terrain harvester")
    ap.add_argument("--source",  choices=["lroc","ohrc","both"], default="both")
    ap.add_argument("--frames",  type=int, default=5,
                    help="Target frames per source (default 5)")
    ap.add_argument("--output",  default="reference_corpus",
                    help="Output directory (default ./reference_corpus)")
    ap.add_argument("--dry-run",   action="store_true",
                    help="List candidates without downloading")
    ap.add_argument("--max-vols", type=int, default=10,
                    help="Max PDS volumes to scan per run (indexes cached, default 10)")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "XenarchHarvester/1.0 (planetary research)"

    all_chips: List[Dict] = []

    if args.source in ("lroc", "both"):
        log.info("═" * 50)
        log.info("Source 1: LRO NAC (pds.lroc.asu.edu)")
        log.info("═" * 50)
        h = NACHarvester(out / "lroc_nac", session, args.frames, args.dry_run, args.max_vols)
        all_chips.extend(h.run())

    if args.source in ("ohrc", "both"):
        log.info("═" * 50)
        log.info("Source 2: Ch-2 OHRC (PRADAN)")
        log.info("═" * 50)
        h = OHRCHarvester(out / "ch2_ohrc", session, args.frames, args.dry_run)
        all_chips.extend(h.run())

    if all_chips:
        write_manifest(out, all_chips)
        log.info(f"\nDone. {len(all_chips)} reference chips in {out}/")
        log.info("Pass to xenarch_pipeline via:")
        log.info(f"  manifest = json.load(open('{out}/manifest.json'))")
        log.info(f"  reference_paths = [c['chip_path'] for c in manifest['chips']]")
    else:
        log.warning("No chips collected.")

if __name__ == "__main__":
    main()
