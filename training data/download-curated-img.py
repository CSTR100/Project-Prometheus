#!/usr/bin/env python3
"""
PDS Imaging Atlas -> download -> inspect -> square-tile -> manifest pipeline.

Source : https://pds-imaging.jpl.nasa.gov/tools/atlas/search  (Atlas IV search API)
Product: MRO CTX EDR and MRO HiRISE EDR (.IMG, PDS3, attached PVL label) -- Mars,
         public domain. Both are 8-bit; CTX uses fixed-length records, HiRISE is an
         unstructured byte stream with byte-offset pointers (see parse_attached_label).
         For HiRISE EDR the Atlas ingest's positional geometry is unreliable, so only
         dimensions/timing/instrument config (attached label) and illumination angles
         are kept -- see the per-row metadata_note.

Per product it:
  1. downloads the raw .IMG via the Atlas data-access API
  2. records product id, url, bytes, UTC download timestamp, SHA-256
  3. parses the attached PVL label (dimensions, record geometry)
  4. pulls the label-derived geometry the Atlas ingest parsed out of the PDS label
     (resolution m/px, incidence, emission, sub-solar azimuth, center lat/lon, ...)
  5. checks for a burned-in watermark (CTX EDR science data has none -> verified, not removed)
  6. summarises illumination (light) geometry
  7. decodes the raster and cuts the long strip into consecutive NxN squares
     (last short block is zero-padded to square and flagged)
Outputs: out/<pid>/tile_###.png, manifest.csv, manifest.parquet, tiles.csv
"""

import sys, os, json, time, hashlib, io, datetime, urllib.request, urllib.parse

import numpy as np
import pvl
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # CTX strips are big; we trust our own input

ATLAS_SEARCH = "https://pds-imaging.jpl.nasa.gov/api/search/atlas/_search"
ATLAS_DATA   = "https://pds-imaging.jpl.nasa.gov/api/data/"          # + atlas: uri
HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "out")

# ---------------------------------------------------------------------------
# Products the pipeline downloads on every run.
#
# Paste MRO PDS Imaging Atlas links below -- either the full "record" page URL
#   https://pds-imaging.jpl.nasa.gov/tools/atlas/record?uri=atlas:pds3:...IMG
# or a bare  atlas:pds3:...IMG  document id. Each entry is fetched, its label
# parsed, its raster cut into square tiles, and a manifest / metadata row added.
# The product id is taken from the .IMG file name.
# ---------------------------------------------------------------------------
SEED_LINKS = [
    # --- MRO HiRISE EDR single-CCD channels, orbits 75490-75499 (8-bit after onboard LUT) ---
    "https://pds-imaging.jpl.nasa.gov/tools/atlas/record?uri=atlas:pds3:mro:mars_reconnaissance_orbiter:/HiRISE/EDR/ESP/ORB_075400_075499/ESP_075499_0935/ESP_075499_0935_BG13_1.IMG",
    "https://pds-imaging.jpl.nasa.gov/tools/atlas/record?uri=atlas:pds3:mro:mars_reconnaissance_orbiter:/HiRISE/EDR/ESP/ORB_075400_075499/ESP_075499_1285/ESP_075499_1285_RED6_1.IMG",
    "https://pds-imaging.jpl.nasa.gov/tools/atlas/record?uri=atlas:pds3:mro:mars_reconnaissance_orbiter:/HiRISE/EDR/ESP/ORB_075400_075499/ESP_075498_1970/ESP_075498_1970_RED6_1.IMG",
    "https://pds-imaging.jpl.nasa.gov/tools/atlas/record?uri=atlas:pds3:mro:mars_reconnaissance_orbiter:/HiRISE/EDR/ESP/ORB_075400_075499/ESP_075498_1610/ESP_075498_1610_RED7_1.IMG",
    "https://pds-imaging.jpl.nasa.gov/tools/atlas/record?uri=atlas:pds3:mro:mars_reconnaissance_orbiter:/HiRISE/EDR/ESP/ORB_075400_075499/ESP_075498_1055/ESP_075498_1055_RED1_0.IMG",
    "https://pds-imaging.jpl.nasa.gov/tools/atlas/record?uri=atlas:pds3:mro:mars_reconnaissance_orbiter:/HiRISE/EDR/ESP/ORB_075400_075499/ESP_075497_1255/ESP_075497_1255_BG12_1.IMG",
    "https://pds-imaging.jpl.nasa.gov/tools/atlas/record?uri=atlas:pds3:mro:mars_reconnaissance_orbiter:/HiRISE/EDR/ESP/ORB_075400_075499/ESP_075497_1255/ESP_075497_1255_RED4_1.IMG",
    "https://pds-imaging.jpl.nasa.gov/tools/atlas/record?uri=atlas:pds3:mro:mars_reconnaissance_orbiter:/HiRISE/EDR/ESP/ORB_075400_075499/ESP_075493_1045/ESP_075493_1045_BG12_1.IMG",
    "https://pds-imaging.jpl.nasa.gov/tools/atlas/record?uri=atlas:pds3:mro:mars_reconnaissance_orbiter:/HiRISE/EDR/ESP/ORB_075400_075499/ESP_075493_1045/ESP_075493_1045_RED0_0.IMG",
    "https://pds-imaging.jpl.nasa.gov/tools/atlas/record?uri=atlas:pds3:mro:mars_reconnaissance_orbiter:/HiRISE/EDR/ESP/ORB_075400_075499/ESP_075492_1760/ESP_075492_1760_RED0_0.IMG",
    "https://pds-imaging.jpl.nasa.gov/tools/atlas/record?uri=atlas:pds3:mro:mars_reconnaissance_orbiter:/HiRISE/EDR/ESP/ORB_075400_075499/ESP_075491_1730/ESP_075491_1730_BG12_1.IMG",
    "https://pds-imaging.jpl.nasa.gov/tools/atlas/record?uri=atlas:pds3:mro:mars_reconnaissance_orbiter:/HiRISE/EDR/ESP/ORB_075400_075499/ESP_075490_1440/ESP_075490_1440_RED4_1.IMG",
    "https://pds-imaging.jpl.nasa.gov/tools/atlas/record?uri=atlas:pds3:mro:mars_reconnaissance_orbiter:/HiRISE/EDR/ESP/ORB_075400_075499/ESP_075490_0920/ESP_075490_0920_IR10_0.IMG",
    "https://pds-imaging.jpl.nasa.gov/tools/atlas/record?uri=atlas:pds3:mro:mars_reconnaissance_orbiter:/HiRISE/EDR/ESP/ORB_075400_075499/ESP_075489_1380/ESP_075489_1380_IR10_0.IMG",
    "https://pds-imaging.jpl.nasa.gov/tools/atlas/record?uri=atlas:pds3:mro:mars_reconnaissance_orbiter:/HiRISE/EDR/ESP/ORB_075400_075499/ESP_075489_1380/ESP_075489_1380_RED7_0.IMG",
]


def _doc_id(link):
    """Accept a full Atlas 'record?uri=...' URL or a bare atlas: doc id."""
    link = link.strip()
    if link.lower().startswith("http"):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
        link = q.get("uri", [""])[0]
    if not link.startswith("atlas:"):
        raise SystemExit(f"cannot parse Atlas link/uri: {link!r}")
    return link


def _product_id(doc_id):
    name = doc_id.rsplit("/", 1)[-1]
    return name[:-4] if name.lower().endswith(".img") else name


# id -> Atlas document id, de-duplicated, order preserved.
PRODUCTS = {}
for _link in SEED_LINKS:
    _doc = _doc_id(_link)
    PRODUCTS.setdefault(_product_id(_doc), _doc)
PRODUCT_IDS = list(PRODUCTS)


def family(product_id):
    """'hirise' or 'ctx' -- the two products differ in label layout and in which
    Atlas-ingest label keys are trustworthy."""
    return "hirise" if "/HiRISE/" in PRODUCTS[product_id] else "ctx"


def http_json(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def atlas_record(product_id):
    """Full Atlas index record for one CTX EDR product id (looked up by doc id)."""
    q = {"size": 1, "query": {"ids": {"values": [PRODUCTS[product_id]]}}}
    hits = http_json(ATLAS_SEARCH, q)["hits"]["hits"]
    if not hits:
        raise SystemExit(f"no Atlas record for {product_id}")
    return hits[0]["_source"]


def download(url, dest):
    t0 = time.time()
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    sha = hashlib.sha256()
    n = 0
    req = urllib.request.Request(url, headers={"User-Agent": "ctx-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            sha.update(chunk)
            n += len(chunk)
    return dict(url=url, bytes=n, download_timestamp=ts,
               sha256=sha.hexdigest(), seconds=round(time.time() - t0, 1))


def parse_attached_label(path):
    """Read the PVL label sitting at the front of a CTX EDR .IMG."""
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    end = head.find(b"\nEND\r\n")
    if end == -1:
        end = head.find(b"\nEND ")
    text = head[: end + 5].decode("latin1")
    label = pvl.loads(text)
    img = label["IMAGE"]
    img_ptr = label["^IMAGE"]
    ptr_units = getattr(img_ptr, "units", None)
    img_rec = int(img_ptr.value if hasattr(img_ptr, "value") else img_ptr)

    if str(ptr_units).upper() == "BYTES" or "RECORD_BYTES" not in label:
        # HiRISE EDR: RECORD_TYPE = UNDEFINED, an unstructured byte stream whose
        # pointers are 1-based BYTE offsets (not record numbers).
        rec_bytes = 1
        lr = label.get("LABEL_RECORDS")
        label_recs = int(getattr(lr, "value", lr) or 0)
        image_start = img_rec - 1
    else:
        # CTX EDR: fixed-length records, ^IMAGE is a 1-based record number.
        rec_bytes = int(label["RECORD_BYTES"])
        label_recs = int(label["LABEL_RECORDS"])
        image_start = (img_rec - 1) * rec_bytes
    return dict(
        lines=int(img["LINES"]),
        line_samples=int(img["LINE_SAMPLES"]),
        sample_bits=int(img["SAMPLE_BITS"]),
        line_prefix_bytes=int(img.get("LINE_PREFIX_BYTES", 0)),
        line_suffix_bytes=int(img.get("LINE_SUFFIX_BYTES", 0)),
        record_bytes=rec_bytes,
        label_records=label_recs,
        image_start_byte=image_start,
        data_offset_source=str(img_ptr),
    )


def load_raster(path, lab):
    """Decode the 8-bit CTX EDR raster into a (lines, samples) uint8 array."""
    assert lab["sample_bits"] == 8, "pipeline handles 8-bit CTX EDR only"
    rows, cols = lab["lines"], lab["line_samples"]
    row_stride = lab["line_prefix_bytes"] + cols + lab["line_suffix_bytes"]
    count = rows * row_stride
    with open(path, "rb") as f:
        f.seek(lab["image_start_byte"])
        buf = np.frombuffer(f.read(count), dtype=np.uint8)
    if buf.size < count:                       # tolerate a short trailing record
        rows = buf.size // row_stride
        buf = buf[: rows * row_stride]
    arr = buf.reshape(rows, row_stride)
    x0 = lab["line_prefix_bytes"]
    return arr[:, x0:x0 + cols]


def watermark_check(arr):
    """CTX EDR (Level 0 science data) carries no burned-in watermark or label bar
    by product spec. Sanity-check anyway: a rendered text/label band would be a
    wide edge block whose brightness sits far from the scene mean with almost no
    texture (std ~ 0). Sensor noise on real data keeps edge std well above that."""
    m = float(arr.mean())
    bands = {"top": arr[:16], "bottom": arr[-16:],
             "left": arr[:, :16], "right": arr[:, -16:]}
    hits = [k for k, b in bands.items()
            if float(b.std()) < 1.0 and abs(float(b.mean()) - m) > 40]
    return dict(watermark_detected=bool(hits),
               watermark_bands=",".join(hits),
               note="CTX EDR science data has no watermark/overlay; edge bands "
                    "checked for a rendered label bar, nothing found, nothing removed"
                    if not hits else "edge band(s) look uniform - inspect: " + ",".join(hits))


def square_tiles(arr, pid):
    """Cut the (tall) strip into consecutive side x side squares (side = width)."""
    rows, cols = arr.shape
    side = min(rows, cols)
    d = os.path.join(OUT, pid)
    os.makedirs(d, exist_ok=True)
    rec = []
    idx = 0
    for y in range(0, rows, side):
        block = arr[y:y + side, :side]
        padded = block.shape[0] < side
        if padded:
            block = np.pad(block, ((0, side - block.shape[0]), (0, 0)))
        p = os.path.join(d, f"tile_{idx:03d}.png")
        Image.fromarray(block, mode="L").save(p)
        rec.append(dict(product_id=pid, tile_index=idx, row_start=y,
                        row_end=min(y + side, rows), side_px=side, padded=padded,
                        path=os.path.relpath(p, HERE),
                        sha256=hashlib.sha256(open(p, "rb").read()).hexdigest()))
        idx += 1
    return side, rec


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def write_metadata_txt(rows, path):
    """Plain-text metadata card per product: name, then the 5 headline facts."""
    def r2(x):
        return round(x, 2) if isinstance(x, (int, float)) else x

    blocks = []
    for r in rows:
        alt = r["spacecraft_altitude_km"]
        alt_s = f"{alt:.2f} km" if alt is not None else "n/a (Atlas geometry inconsistent for this product)"
        inc = r["incidence_angle_deg"]
        sun_elev = f"{90 - inc:.1f} deg" if inc is not None else "n/a"
        caveat = "" if alt is not None else "  [Atlas EDR values, see metadata_note]"
        light_s = (
            f"incidence {r2(inc)} deg, emission {r2(r['emission_angle_deg'])} deg, "
            f"phase {r2(r['phase_angle_deg'])} deg; sub-solar azimuth "
            f"{r2(r['sub_solar_azimuth_deg'])} deg; solar longitude (Ls) "
            f"{r2(r['solar_longitude_deg'])} deg; local solar time "
            f"{r2(r['local_solar_time_h'])} h  ->  daytime, sun ~{sun_elev} above horizon{caveat}"
        )
        blocks.append(
            f"=== {r['product_id']} ===\n"
            f"image                : {os.path.basename(r['local_img_path'])}  ({r['instrument']} EDR)\n"
            f"1) watermark detected : {r['watermark_detected']}\n"
            f"2) watermark removed  : {r['watermark_removed']}\n"
            f"3) altitude           : {alt_s}\n"
            f"4) resolution         : {r['resolution_px']} px\n"
            f"5) light condition    : {light_s}\n"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, tile_rows = [], []
    for pid in PRODUCT_IDS:
        print(f"\n=== {pid} ===")
        src = atlas_record(pid)
        L = src.get("pds3_label", {})
        uri = src["gather"]["pds_archive"]["related"]["src"]["uri"]
        url = ATLAS_DATA + urllib.parse.quote(uri, safe=":/")
        img_path = os.path.join(OUT, pid + ".IMG")

        dl = download(url, img_path)
        print(f"  downloaded {dl['bytes']:,} B in {dl['seconds']}s  sha256={dl['sha256'][:16]}...")

        lab = parse_attached_label(img_path)
        arr = load_raster(img_path, lab)
        wm = watermark_check(arr)
        side, trec = square_tiles(arr, pid)
        tile_rows += trec
        n_full = sum(1 for t in trec if not t["padded"])
        n_pad = sum(1 for t in trec if t["padded"])
        print(f"  raster {arr.shape[1]}x{arr.shape[0]}  ->  {len(trec)} squares of {side}px "
              f"({n_full} full, {n_pad} padded)")

        fam = family(pid)
        pick = lambda *ks: next((L[k] for k in ks if L.get(k) is not None), None)

        # Dimensions always come from the attached PVL label -- for HiRISE EDR the
        # Atlas ingest's SAMPLES/LINES/SAMPLE_BITS are column-shifted and unusable.
        img_w = int(pick("imageSamples") or lab["line_samples"])
        img_h = int(pick("imageLines") or lab["lines"])
        resolution_px = f"{img_w}x{img_h}"          # pixel dimensions, W x H
        res = num(pick("scaledPixelWidth", "SCALED_PIXEL_WIDTH"))
        light = dict(
            incidence_angle_deg=num(pick("incidenceAngle", "INCIDENCE_ANGLE")),
            emission_angle_deg=num(pick("emissionAngle", "EMISSION_ANGLE")),
            phase_angle_deg=num(pick("phaseAngle", "PHASE_ANGLE")),
            sub_solar_azimuth_deg=num(pick("subSolarAzimuth", "SUB_SOLAR_AZIMUTH")),
            north_azimuth_deg=num(pick("northAzimuth", "NORTH_AZIMUTH")),
            solar_longitude_deg=num(pick("solarLongitude", "SOLAR_LONGITUDE")),
            local_solar_time_h=num(pick("localTime", "LOCAL_TIME")),
            sub_solar_latitude_deg=num(pick("subSolarLatitude", "SUB_SOLAR_LATITUDE")),
        )
        print("  light: inc=%s  em=%s  phase=%s  subSolarAz=%s  Ls=%s  LST=%sh"
              % (light["incidence_angle_deg"], light["emission_angle_deg"],
                 light["phase_angle_deg"], light["sub_solar_azimuth_deg"],
                 light["solar_longitude_deg"], light["local_solar_time_h"]))
        print(f"  resolution = {resolution_px} px  ({res} m/pixel from label)")

        if fam == "hirise":
            # HiRISE EDR (raw single-CCD channel). The Atlas ingest's positional
            # geometry for these products is internally inconsistent
            # (center lat/lon, altitude, target/slant distance, lat/lon bounds all
            # disagree), so those are dropped; illumination angles are kept.
            center_lat = center_lon = None
            spacecraft_altitude_km = scaled_w_km = scaled_h_km = None
            footprint = None
            instrument = "MRO HiRISE"
            coord_ref = ("Mars IAU 2000 areocentric; HiRISE EDR - raw single-CCD "
                         "channel, no map projection")
            metadata_note = ("dimensions/timing/instrument config from attached PVL "
                             "label; Atlas-ingest positional geometry inconsistent, "
                             "omitted; illumination angles retained from Atlas label")
        else:
            center_lat = num(pick("centerLatitude"))
            center_lon = num(pick("centerLongitude"))
            spacecraft_altitude_km = num(pick("spacecraftAltitude"))
            scaled_w_km = num(pick("scaledImageWidth"))
            scaled_h_km = num(pick("scaledImageHeight"))
            footprint = json.dumps({
                "ul": [num(L.get("upperLeftLatitude")),  num(L.get("upperLeftLongitude"))],
                "ur": [num(L.get("upperRightLatitude")), num(L.get("upperRightLongitude"))],
                "ll": [num(L.get("lowerLeftLatitude")),  num(L.get("lowerleftLongitude"))],
                "lr": [num(L.get("lowerRightLatitude")), num(L.get("lowerRightLongitude"))],
            })
            instrument = "MRO CTX"
            coord_ref = "Mars IAU 2000 areocentric; CTX EDR Level 0 - no map projection"
            metadata_note = None

        row = dict(
            product_id=pid,
            url=url,
            bytes=dl["bytes"],
            download_timestamp=dl["download_timestamp"],
            sha256=dl["sha256"],
            # ---- parsed from the PDS label ----
            resolution_m_per_px=res,
            resolution_px=resolution_px,
            incidence_angle_deg=light["incidence_angle_deg"],
            emission_angle_deg=light["emission_angle_deg"],
            sub_solar_azimuth_deg=light["sub_solar_azimuth_deg"],
            center_lat=center_lat,
            center_lon=center_lon,
            image_width_px=img_w,
            image_height_px=img_h,
            dimension=resolution_px,
            light=json.dumps(light),
            extradata_reference=uri,
            # ---- extra label / context fields ----
            phase_angle_deg=light["phase_angle_deg"],
            north_azimuth_deg=light["north_azimuth_deg"],
            solar_longitude_deg=light["solar_longitude_deg"],
            local_solar_time_h=light["local_solar_time_h"],
            sub_solar_latitude_deg=light["sub_solar_latitude_deg"],
            spacecraft_altitude_km=spacecraft_altitude_km,
            scaled_image_width_km=scaled_w_km,
            scaled_image_height_km=scaled_h_km,
            emission_note="near-nadir" if (light["emission_angle_deg"] or 9) < 5 else "off-nadir",
            image_time=pick("imageTime", "START_TIME"),
            orbit_number=pick("orbitNumber", "ORBIT_NUMBER"),
            mission_phase=pick("missionPhaseName", "MISSION_PHASE_NAME"),
            data_set_id=pick("dsId", "DATA_SET_ID"),
            instrument=instrument,
            target="MARS",
            coordinate_reference=coord_ref,
            metadata_note=metadata_note,
            footprint_corners_lat_lon=footprint,
            label_record_bytes=lab["record_bytes"],
            label_records=lab["label_records"],
            image_start_byte=lab["image_start_byte"],
            watermark_detected="yes" if wm["watermark_detected"] else "no",
            watermark_removed="no",  # pipeline never strips pixels; CTX EDR carries no overlay
            watermark_note=wm["note"],
            square_tile_side_px=side,
            n_square_tiles=len(trec),
            n_tiles_padded=n_pad,
            download_seconds=dl["seconds"],
            local_img_path=os.path.relpath(img_path, HERE),
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "manifest.csv"), index=False)
    try:
        df.to_parquet(os.path.join(HERE, "manifest.parquet"), index=False)
    except Exception as e:                       # pragma: no cover
        print("parquet skipped:", e)
    pd.DataFrame(tile_rows).to_csv(os.path.join(HERE, "tiles.csv"), index=False)
    write_metadata_txt(rows, os.path.join(HERE, "metadata.txt"))

    res_series = df["resolution_m_per_px"].dropna()
    avg = res_series.mean()
    print("\n---------------------------------------------")
    for _, r in df.iterrows():
        print(f"{r['product_id']}: {r['resolution_m_per_px']} m/px")
    print(f"AVERAGE RESOLUTION (n={len(res_series)}): {avg:.3f} m/pixel")
    print("manifest.csv / manifest.parquet / tiles.csv written to", HERE)


if __name__ == "__main__":
    main()
