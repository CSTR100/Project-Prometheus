#!/usr/bin/env python3
"""
Multi-dataset planetary imagery downloader.

Covers the sources shared in chat:

  # HiRISE EDR (Mars Reconnaissance Orbiter)
  #   Browse UI : https://pds-imaging.jpl.nasa.gov/tools/atlas/search?gather.common.instrument=HIRISE&gather.common.product_type=EDR
  #   API used  : PDS Imaging Atlas Solr search  https://pds-imaging.jpl.nasa.gov/solr/pds_archives/search
  #   Downloads : raw .IMG files from https://hirise-pds.lpl.arizona.edu/PDS/<FILE_NAME_SPECIFICATION>
  #   Status    : verified working (see download_hirise_edr.py, the dedicated script for this one)

  # CTX (Context Camera, Mars Reconnaissance Orbiter) -- "low quality but ok"
  #   Browse UI : https://pds-imaging.jpl.nasa.gov/tools/atlas/search?gather.common.mission=mro&gather.common.instrument=CTX
  #   API used  : same PDS Imaging Atlas Solr search as HiRISE, filtered to instrument=ctx
  #   Downloads : constructed from ATLAS_VOLUME_URL + FILE_PATH + FILE_NAME_SPECIFICATION
  #   Status    : UNVERIFIED -- several sampled products 404'd through the pds-imaging.jpl.nasa.gov
  #               mirror at test time (2026-08-27). CTX EDRs may have moved to a different mirror
  #               (e.g. the PDS Geosciences Node or MSSS). Check ATLAS_DATA_URL / ATLAS_LABEL_URL
  #               in the returned doc and adjust build_url() below if downloads keep failing.

  # LRO (Lunar Reconnaissance Orbiter) -- "kinda" -- using the LROC camera instrument
  #   Browse UI : https://pds-imaging.jpl.nasa.gov/tools/atlas/search?gather.common.mission=lro
  #   API used  : same Solr search, filtered to instrument=lroc (the plain "mission=lro" filter
  #               also matches non-imagery instruments like LAMP, which is why LROC is pinned here)
  #   Downloads : ATLAS_VOLUME_URL + "/" + FILE_SPECIFICATION_NAME, following redirects
  #               (redirects through lroc.sese.asu.edu -> lroc.im-ldi.com -> pds.mcp.nasa.gov)
  #   Status    : verified working, but LROC NAC frames are large (~500 MB each) -- the disk
  #               guard below matters a lot more here than for the other datasets.

  # MOC (Mars Orbiter Camera, Mars Global Surveyor) -- "low res"
  #   Browse UI : https://pds-imaging.jpl.nasa.gov/tools/atlas/search?gather.common.spacecraft=mars_global_surveyor&gather.common.instrument=MOC
  #   API used  : same Solr search, filtered to spacecraft=mars global surveyor, instrument=moc
  #   Downloads : https://pds-imaging.jpl.nasa.gov/data/mgs/moc/ + FILE_PATH + FILE_NAME
  #               (compressed .imq images -- need MOC/ISIS tools to decompress into a raster)
  #   Status    : verified working

  # Chandrayaan-2 TMC2 (ISRO Pradan)
  #   Browse UI : https://pradan.issdc.gov.in/ch2/protected/browse.xhtml?id=tmc2
  #             : https://pradan.issdc.gov.in/ch2/protected/browse.xhtml
  #   API used  : NONE -- this is a JSF (JavaServer Faces) portal behind an ISSDC login.
  #               There is no public/documented REST API; the "protected" path in the URL
  #               means every request needs an authenticated session (cookies + a JSF
  #               viewstate/CSRF token generated per session). Scripting this requires
  #               logging in through a real browser first, then replaying the session
  #               cookie -- it cannot be done anonymously like the NASA PDS endpoints above.
  #   Status    : NOT IMPLEMENTED. See download_pradan_tmc2() below for what's needed.

All NASA/PDS datasets default to sampling ~1% of the archive (systematic sampling,
1 product every `stride` results) rather than downloading sequential blocks, and all
downloads stop automatically if free disk space drops below --min-free-mb.

This script only defines the downloader -- run it explicitly to actually fetch data:
    python download_all_datasets.py --dataset hirise --pct 1.0
    python download_all_datasets.py --dataset all --pct 1.0 --outdir data
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
import urllib.request
import urllib.parse

SOLR_URL = "https://pds-imaging.jpl.nasa.gov/solr/pds_archives/search"
USER_AGENT = "planetary-dataset-sample-downloader/1.0 (contact: giulia.sironi.02@gmail.com)"


# --- per-dataset configuration ------------------------------------------------
#
# `fq` are the Solr filter-query clauses that select this instrument/mission in the
# PDS Imaging Atlas index. `build_url(doc)` turns one Solr result document into the
# actual file URL to download, using the fields observed in that dataset's documents.

def _hirise_url(doc):
    # hirise-pds.lpl.arizona.edu hosts self-labeled PDS3 .IMG files directly.
    return "https://hirise-pds.lpl.arizona.edu/PDS/" + doc["FILE_NAME_SPECIFICATION"]


def _ctx_url(doc):
    # Best-effort reconstruction from the Atlas record; see the CTX note above.
    return doc["ATLAS_VOLUME_URL"] + "/" + doc["FILE_PATH"] + "/" + doc["FILE_NAME_SPECIFICATION"]


def _moc_url(doc):
    return "https://pds-imaging.jpl.nasa.gov/data/mgs/moc/" + doc["FILE_PATH"] + doc["FILE_NAME"]


def _lroc_url(doc):
    return doc["ATLAS_VOLUME_URL"] + "/" + doc["FILE_SPECIFICATION_NAME"]


DATASETS = {
    "hirise": {
        "label": "HiRISE EDR (Mars Reconnaissance Orbiter)",
        "fq": ["ATLAS_INSTRUMENT_NAME:hirise", "PRODUCT_TYPE:edr"],
        "build_url": _hirise_url,
        "filename_field": "FILE_NAME_SPECIFICATION",
    },
    "ctx": {
        "label": "CTX (Mars Reconnaissance Orbiter)",
        "fq": ["ATLAS_INSTRUMENT_NAME:ctx", 'ATLAS_MISSION_NAME:"Mars Reconnaissance Orbiter"'],
        "build_url": _ctx_url,
        "filename_field": "FILE_NAME_SPECIFICATION",
    },
    "moc": {
        "label": "MOC (Mars Global Surveyor)",
        "fq": ["ATLAS_INSTRUMENT_NAME:moc", 'ATLAS_SPACECRAFT_NAME:"Mars Global Surveyor"'],
        "build_url": _moc_url,
        "filename_field": "FILE_NAME",
    },
    "lroc": {
        "label": "LROC (Lunar Reconnaissance Orbiter)",
        "fq": ["ATLAS_INSTRUMENT_NAME:lroc"],
        "build_url": _lroc_url,
        "filename_field": "FILE_SPECIFICATION_NAME",
    },
}


# --- Solr search helpers -------------------------------------------------------

def solr_query(fq, count, start=0):
    params = {"q": "*:*", "fq": fq, "rows": str(count), "start": str(start), "wt": "json"}
    qs = urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(f"{SOLR_URL}?{qs}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return data["response"]["docs"]


def solr_num_found(fq):
    params = {"q": "*:*", "fq": fq, "rows": "0", "wt": "json"}
    qs = urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(f"{SOLR_URL}?{qs}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return data["response"]["numFound"]


def iter_sample_docs(fq, pct):
    """Systematic sample: pct% of the archive, spread evenly (1 product every `stride`)."""
    total = solr_num_found(fq)
    target = max(1, int(total * pct / 100))
    stride = max(1, total // target)
    print(f"    archive size: {total} products, sampling {pct}% => ~{target} products "
          f"(1 every {stride})")
    offset = 0
    yielded = 0
    while offset < total and yielded < target:
        docs = solr_query(fq, 1, start=offset)
        if docs:
            yield docs[0]
            yielded += 1
        offset += stride


# --- download helpers -----------------------------------------------------------

def free_mb(path):
    return shutil.disk_usage(path).free / (1024 * 1024)


def download_file(url, dest_path, retries=3):
    if os.path.exists(dest_path):
        print(f"    already present, skipping: {os.path.basename(dest_path)}")
        return True
    tmp_path = dest_path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp, open(tmp_path, "wb") as f:
                total = resp.getheader("Content-Length")
                total = int(total) if total else None
                downloaded = 0
                chunk = 1024 * 256
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    downloaded += len(buf)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"\r    {os.path.basename(dest_path)}: {pct}% "
                              f"({downloaded}/{total} bytes)", end="")
            print()
            os.replace(tmp_path, dest_path)
            return True
        except Exception as e:
            print(f"\n    attempt {attempt}/{retries} failed: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            time.sleep(2 * attempt)
    return False


def download_pradan_tmc2(*_args, **_kwargs):
    """Chandrayaan-2 TMC2 browse data (ISSDC Pradan) -- NOT IMPLEMENTED.

    Pradan (https://pradan.issdc.gov.in/ch2/protected/browse.xhtml?id=tmc2) sits behind
    a login-gated JSF portal, not a public API. To make this work you would need to:
      1. Log in through a real browser with an ISSDC account.
      2. Capture the session cookie(s) and JSF viewstate/CSRF token from a logged-in
         request (browser dev tools -> Network tab on a search/download request).
      3. Replay that session (cookies + viewstate) in this script's requests, refreshing
         it as it expires -- Pradan sessions are typically short-lived.
    There is no equivalent of the NASA PDS Solr search here, so query parameters and
    result parsing would have to be reverse-engineered from the portal's own requests.
    """
    raise NotImplementedError(
        "Pradan/Chandrayaan-2 requires an authenticated ISSDC session; see the "
        "docstring of download_pradan_tmc2() for what's needed before this can run."
    )


# --- main -------------------------------------------------------------------------

def run_dataset(name, pct, outdir, delay, min_free_mb):
    cfg = DATASETS[name]
    print(f"\n=== {cfg['label']} ===")
    manifest_path = os.path.join(outdir, "manifest.csv")
    write_header = not os.path.exists(manifest_path)

    ok, failed, stopped_for_space = 0, 0, 0
    with open(manifest_path, "a", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        if write_header:
            writer.writerow(["dataset", "product_id", "filename", "download_url", "local_path", "status"])
        for doc in iter_sample_docs(cfg["fq"], pct):
            free = free_mb(outdir)
            if free < min_free_mb:
                print(f"    free space dropped to {free:.0f} MB (< {min_free_mb} MB), stopping.")
                stopped_for_space += 1
                break
            fname_field = cfg["filename_field"]
            if fname_field not in doc:
                continue
            try:
                url = cfg["build_url"](doc)
            except KeyError as e:
                print(f"    skipping product, missing field {e}")
                continue
            fname = os.path.basename(doc[fname_field])
            dest = os.path.join(outdir, name, fname)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            print(f"  [{name}] {fname} (free space: {free:.0f} MB)")
            success = download_file(url, dest)
            status = "OK" if success else "FAILED"
            ok += success
            failed += not success
            writer.writerow([name, doc.get("PRODUCT_ID", ""), fname, url,
                              dest if success else "", status])
            mf.flush()
            time.sleep(delay)

    print(f"  {name}: {ok} downloaded, {failed} failed"
          f"{', stopped for low disk space' if stopped_for_space else ''}.")


def main():
    ap = argparse.ArgumentParser(description="Download samples from multiple planetary imagery datasets")
    ap.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all",
                     help="which dataset to sample (default: all)")
    ap.add_argument("--pct", type=float, default=1.0,
                     help="percent of each dataset's archive to sample (default 1.0 = 1%%)")
    ap.add_argument("--outdir", default="data", help="destination folder")
    ap.add_argument("--delay", type=float, default=1.0, help="pause in seconds between downloads")
    ap.add_argument("--min-free-mb", type=float, default=300,
                     help="abort downloads if free disk space drops below this threshold (MB)")
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        run_dataset(name, args.pct, outdir, args.delay, args.min_free_mb)


if __name__ == "__main__":
    main()
