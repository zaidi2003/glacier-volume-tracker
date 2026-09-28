#!/usr/bin/env python3
"""Monthly snow/ice surface area + one static glacier outline, over a fixed AOI.

Stages (``python3 glacier_mask_area.py [fetch|outline|mask|report|all]``):

  fetch   Pull a 5-band L2A stack (B02, B03, B04, B08, B11) for every month of
          YEAR over a 15 x 15 km box in UTM 43N.  At 10 m that is exactly
          1500 x 1500 px, so one pixel = 100 m2 and area math is exact.
          Cached in stacks/*.npz.

  outline Build the reference glacier outline ONCE with prompted SAM2: a fixed
          set of positive spine points on each of the two glacier branches
          (clean ice trunk, debris-covered eastern branch) plus negative points
          on the rock interfluve.  Prompts were placed by eye on a coordinate
          grid; result verified visually in glacier_masks/outline_overlay.png.

  glacier Re-run that SAME prompted segmentation on every monthly image:
          one encoder pass per month, then the identical prompts are decoded
          against that month's appearance.  Saves glacier_masks/*_glacier.png
          per month, so each month gets its own AI boundary (snow-covered ice,
          clouds and illumination shift it slightly) instead of a reused one.

  mask    Monthly snow+ice extent - NDSI > 0.40 with an NDWI water cut - drawn
          as a filled translucent yellow, plus that month's prompted-SAM
          glacier boundaries (red = trunk, cyan = east branch).  Writes the
          filled overlays for easy manual checking.

Honest reading guide: month-to-month movement of the glacier boundary mostly
reflects segmentation/coverage conditions (snow on ice, clouds, shadows), not
real glacier change - the glacier does not measurably advance/retreat within
one year at 10 m.  The NDSI series is the physical signal.

  report  glacier_areas.csv + glacier_area_2024.png + persistent masks
          (classified as snow/ice in >= 9 of 12 months).

Known limits: clouds shadow the index slightly (thin cirrus passes), and
debris-covered ice is only in the static outline, never in the monthly series.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- parameters
YEAR = 2024
CENTER_LAT = 36.149568
CENTER_LON = 74.856459
UTM_EPSG = "EPSG:32643"  # UTM zone 43N covers 72-78E, our AOI is 74.86E
SIZE_M = 15_000
RES_M = 10
BANDS = ("B02", "B03", "B04", "B08", "B11")
PIXEL_M2 = RES_M * RES_M  # 100 m2, exact: the grid is defined in UTM metres

CDSE_BASE_URL = "https://sh.dataspace.copernicus.eu"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)

NDSI_SNOW = 0.40  # classic snow/ice threshold (gravity centre of the index)
NDWI_WATER = 0.15  # water has high NDWI; NDWI > this is a lake/river, not snow
MIN_BLOB_PX = 500  # 0.05 km2: smaller specks are index noise
PERSIST_MIN_MONTHS = 9

# Prompted-SAM outline: spine of each branch + negatives on the rock wedge
# between them, read off glacier_masks/grid_september.png (100 px grid).
OUTLINE_LEFT = [
    (490, 50), (520, 200), (560, 400), (600, 600),
    (605, 800), (605, 1000), (610, 1200), (620, 1450),
]
OUTLINE_RIGHT = [
    (740, 690), (830, 760), (910, 870), (975, 1000),
    (1000, 1130), (1015, 1270), (1040, 1420),
]
OUTLINE_NEG = [(770, 830), (790, 1050), (770, 1300), (850, 1480), (880, 850)]
OUTLINE_IMAGE_MONTH = 9  # prompts were drawn on this month's RGB

ROOT = Path(__file__).resolve().parent
STACK_DIR = ROOT / "stacks"
MASK_DIR = ROOT / "glacier_masks"
WEIGHTS = ROOT / "weights" / "sam2.1_hiera_small.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
OUTLINE_PNG = MASK_DIR / "glacier_outline.png"
OUTLINE_VIEW = MASK_DIR / "outline_overlay.png"
CSV_PATH = ROOT / "glacier_areas.csv"
PLOT_PATH = ROOT / "glacier_area_2024.png"

EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B02", "B03", "B04", "B08", "B11"],
            units: "REFLECTANCE"
        }],
        output: { bands: 5, sampleType: "FLOAT32" }
    };
}
function evaluatePixel(s) {
    return [s.B02, s.B03, s.B04, s.B08, s.B11];
}
"""

MONTHS = range(1, 13)


# --------------------------------------------------------------------- fetch
def make_config():
    from dotenv import load_dotenv
    from sentinelhub import SHConfig

    load_dotenv(ROOT / ".env")
    config = SHConfig()
    config.sh_client_id = os.environ.get("SH_CLIENT_ID")
    config.sh_client_secret = os.environ.get("SH_CLIENT_SECRET")
    if not config.sh_client_id or not config.sh_client_secret:
        raise SystemExit("Set SH_CLIENT_ID / SH_CLIENT_SECRET in .env first.")
    config.sh_base_url = CDSE_BASE_URL
    config.sh_token_url = CDSE_TOKEN_URL
    return config


def cdse_collection(config):
    from sentinelhub import DataCollection

    try:
        return DataCollection["cdse_s2l2a"]
    except KeyError:
        return DataCollection.SENTINEL2_L2A.define_from(
            "cdse_s2l2a", service_url=CDSE_BASE_URL
        )


def aoi():
    """15 x 15 km box in UTM 43N -> (bbox, (width, height))."""
    from pyproj import Transformer
    from sentinelhub import CRS, BBox, bbox_to_dimensions

    x, y = Transformer.from_crs("EPSG:4326", UTM_EPSG, always_xy=True).transform(
        CENTER_LON, CENTER_LAT
    )
    h = SIZE_M / 2
    bbox = BBox([x - h, y - h, x + h, y + h], crs=CRS(UTM_EPSG))
    size = bbox_to_dimensions(bbox, resolution=RES_M)
    if size != (SIZE_M // RES_M, SIZE_M // RES_M):
        raise SystemExit(f"unexpected raster size {size}, expected 1500 x 1500")
    return bbox, size


def fetch(months=MONTHS, force=False):
    from sentinelhub import MimeType, SentinelHubRequest

    config = make_config()
    collection = cdse_collection(config)
    bbox, size = aoi()
    STACK_DIR.mkdir(exist_ok=True)

    for month in months:
        out = STACK_DIR / f"{YEAR}_{month:02d}.npz"
        if out.exists() and not force:
            print(f"[{month:02d}] cached {out.name}")
            continue

        start = f"{YEAR}-{month:02d}-01"
        last = calendar.monthrange(YEAR, month)[1]
        end = f"{YEAR}-{month:02d}-{last:02d}"

        request = SentinelHubRequest(
            evalscript=EVALSCRIPT,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=collection,
                    time_interval=(start, end),
                    mosaicking_order="leastCC",
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=size,
            config=config,
        )
        raw = request.get_data()[0]
        if isinstance(raw, (bytes, bytearray)):  # defensive: not decoded upstream
            import tifffile

            raw = tifffile.imread(io.BytesIO(raw))
        data = np.asarray(raw, dtype=np.float32).squeeze()
        if data.ndim != 3 or data.shape[2] != len(BANDS):
            raise SystemExit(f"[{month:02d}] bad stack shape {data.shape}")

        np.savez_compressed(out, data=data.astype(np.float16), rgb=to_rgb(data))
        print(
            f"[{month:02d}] {calendar.month_name[month]:<10} shape={data.shape} "
            f"green_p50={np.median(data[..., 1]):.3f} swir_p50={np.median(data[..., 4]):.3f} "
            f"-> {out.name}"
        )


def to_rgb(data: np.ndarray) -> np.ndarray:
    """B04/B03/B02 with a robust 2-98 percentile stretch, for display."""
    rgb = data[..., [2, 1, 0]]
    out = np.zeros(rgb.shape, dtype=np.uint8)
    for i in range(3):
        band = rgb[..., i]
        valid = band[band > 0]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, [2, 98])
        if hi <= lo:
            hi = lo + 1e-3
        out[..., i] = np.clip((band - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return out


# ------------------------------------------------------------------- indices
def indices(data: np.ndarray):
    green = data[..., 1].astype(np.float32)
    nir = data[..., 3].astype(np.float32)
    swir = data[..., 4].astype(np.float32)
    ndsi = (green - swir) / (green + swir + 1e-6)
    ndwi = (green - nir) / (green + nir + 1e-6)
    return np.nan_to_num(ndsi, nan=-1.0), np.nan_to_num(ndwi, nan=-1.0)


def snow_mask(data: np.ndarray) -> np.ndarray:
    ndsi, ndwi = indices(data)
    return cleanup((ndsi > NDSI_SNOW) & (ndwi < NDWI_WATER))


def cleanup(mask: np.ndarray) -> np.ndarray:
    """Morphological open/close, then drop blobs smaller than MIN_BLOB_PX."""
    mask_u8 = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n > 1:
        keep = np.isin(
            labels, np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= MIN_BLOB_PX) + 1
        )
        mask_u8 = (labels * keep).astype(np.uint8)
    return mask_u8.astype(bool)


def _has_mps() -> bool:
    import torch

    return torch.backends.mps.is_available()


def _load_stack(month: int):
    path = STACK_DIR / f"{YEAR}_{month:02d}.npz"
    if not path.exists():
        raise SystemExit(f"missing {path.name}, run fetch first")
    z = np.load(path)
    return z["data"].astype(np.float32), z["rgb"]


# ------------------------------------------------------------------ outline
def build_predictor(device: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not WEIGHTS.exists():
        raise SystemExit(f"missing weights: {WEIGHTS}")
    model = build_sam2(SAM2_CFG, str(WEIGHTS), device=device)
    print(f"SAM2 ready on {device}")
    return SAM2ImagePredictor(model)


def _prompted_branch(predictor, points, negatives) -> np.ndarray:
    coords = np.array(points + negatives, np.float32)
    labels = np.array([1] * len(points) + [0] * len(negatives), np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=coords, point_labels=labels, multimask_output=True
    )
    best = masks[int(np.argmax(scores))]  # highest SAM confidence won visually
    # keep only components that actually contain a prompt point
    n, lab = cv2.connectedComponents(best.astype(np.uint8))
    keep = {lab[y, x] for x, y in points} - {0}
    return np.isin(lab, list(keep))


def outline_stage(device=None, force=False) -> np.ndarray:
    """Build (once) and cache the static glacier outline."""
    if OUTLINE_PNG.exists() and not force:
        return cv2.imread(str(OUTLINE_PNG), 0) > 0

    MASK_DIR.mkdir(exist_ok=True)
    device = device or ("mps" if _has_mps() else "cpu")
    data, rgb = _load_stack(OUTLINE_IMAGE_MONTH)
    predictor = build_predictor(device)
    predictor.set_image(rgb)
    try:
        left = _prompted_branch(predictor, OUTLINE_LEFT, OUTLINE_NEG)
        right = _prompted_branch(predictor, OUTLINE_RIGHT, OUTLINE_NEG)
    except RuntimeError as exc:
        if "mps" not in str(exc).lower() and "MPS" not in str(exc):
            raise
        print(f"MPS failed ({exc}); retrying outline on CPU")
        predictor = build_predictor("cpu")
        predictor.set_image(rgb)
        left = _prompted_branch(predictor, OUTLINE_LEFT, OUTLINE_NEG)
        right = _prompted_branch(predictor, OUTLINE_RIGHT, OUTLINE_NEG)

    outline = np.logical_or(left, right)
    cv2.imwrite(str(OUTLINE_PNG), outline.astype(np.uint8) * 255)

    canvas = rgb.copy()
    for m, col in ((left, (255, 60, 60)), (right, (60, 255, 255))):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cs, -1, col, 3)
    for x, y in OUTLINE_NEG:
        cv2.circle(canvas, (x, y), 9, (0, 0, 255), -1)
    from PIL import Image

    Image.fromarray(canvas).save(OUTLINE_VIEW)
    print(
        f"outline: trunk {left.sum() * PIXEL_M2 / 1e6:.3f} + east "
        f"{right.sum() * PIXEL_M2 / 1e6:.3f} = {outline.sum() * PIXEL_M2 / 1e6:.3f} km2 "
        f"(see {OUTLINE_VIEW.name})"
    )
    return outline


# ------------------------------------------------------------------ glacier
def glacier_monthly_stage(months=MONTHS, device=None):
    """Prompted SAM2 on every monthly image with the *same* verified prompts.

    Returns {month: {"left": ndarray, "right": ndarray, "mask": ndarray}} and
    writes glacier_masks/YYYY_MM_glacier.png per month.
    """
    MASK_DIR.mkdir(exist_ok=True)
    device = device or ("mps" if _has_mps() else "cpu")
    predictor = build_predictor(device)
    results = {}

    for month in months:
        _, rgb = _load_stack(month)
        try:
            predictor.set_image(rgb)
            left = _prompted_branch(predictor, OUTLINE_LEFT, OUTLINE_NEG)
            right = _prompted_branch(predictor, OUTLINE_RIGHT, OUTLINE_NEG)
        except RuntimeError as exc:  # MPS hiccup -> whole stage on CPU
            if "mps" not in device and "MPS" not in str(exc):
                raise
            print(f"  [{month:02d}] MPS failed ({exc}); switching to CPU")
            predictor = build_predictor("cpu")
            predictor.set_image(rgb)
            left = _prompted_branch(predictor, OUTLINE_LEFT, OUTLINE_NEG)
            right = _prompted_branch(predictor, OUTLINE_RIGHT, OUTLINE_NEG)

        mask = np.logical_or(left, right)
        name = f"{YEAR}_{month:02d}"
        cv2.imwrite(str(MASK_DIR / f"{name}_glacier.png"), mask.astype(np.uint8) * 255)
        results[month] = {"left": left, "right": right, "mask": mask}
        print(
            f"  [{month:02d}] {calendar.month_name[month]:<10} glacier = "
            f"{mask.sum() * PIXEL_M2 / 1e6:7.3f} km2  "
            f"(trunk {left.sum() * PIXEL_M2 / 1e6:.3f} + east "
            f"{right.sum() * PIXEL_M2 / 1e6:.3f})"
        )
    return results


# ---------------------------------------------------------------------- mask
def mask_stage(months=MONTHS, glacier=None):
    """Filled snow/ice overlays annotated with that month's SAM boundaries."""
    MASK_DIR.mkdir(exist_ok=True)
    areas, glacier_areas = {}, {}

    for month in months:
        data, rgb = _load_stack(month)
        snow = snow_mask(data)
        areas[month] = snow

        name = f"{YEAR}_{month:02d}"
        cv2.imwrite(str(MASK_DIR / f"{name}_snowice.png"), snow.astype(np.uint8) * 255)
        ndsi, _ = indices(data)
        cv2.imwrite(
            str(MASK_DIR / f"{name}_ndsi.png"),
            np.clip((ndsi + 1) / 2 * 255, 0, 255).astype(np.uint8),
        )

        # this month's own SAM boundaries (red = trunk, cyan = east branch)
        parts = glacier.get(month) if glacier else None
        if parts is None:  # fall back to a previously saved per-month mask
            saved = cv2.imread(str(MASK_DIR / f"{name}_glacier.png"), 0)
            parts = {"left": None, "right": None, "mask": saved > 0} if saved is not None else None

        canvas = rgb.astype(np.float32)
        canvas[snow] = 0.55 * canvas[snow] + 0.45 * np.array([255, 215, 0], np.float32)
        canvas = canvas.astype(np.uint8)
        glacier_km2 = None
        if parts and parts["mask"] is not None:
            glacier_areas[month] = parts["mask"]
            glacier_km2 = parts["mask"].sum() * PIXEL_M2 / 1e6
            layers = []
            if parts.get("left") is not None:
                layers.append((parts["left"], (255, 60, 60)))
            if parts.get("right") is not None:
                layers.append((parts["right"], (60, 255, 255)))
            if not layers:  # saved combined mask only -> single colour
                layers.append((parts["mask"], (255, 60, 60)))
            for m, col in layers:
                cs, _ = cv2.findContours(
                    m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(canvas, cs, -1, col, 3)

        label = f"{calendar.month_name[month]} {YEAR}   snow+ice {snow.sum() * PIXEL_M2 / 1e6:.2f} km2"
        if glacier_km2 is not None:
            label += f"   glacier {glacier_km2:.2f} km2"
        cv2.putText(
            canvas, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        from PIL import Image

        Image.fromarray(canvas).save(MASK_DIR / f"{name}_overlay.png")
        print(
            f"  [{month:02d}] {calendar.month_name[month]:<10} "
            f"snow+ice = {snow.sum() * PIXEL_M2 / 1e6:7.3f} km2"
            + (
                f"   glacier = {glacier_km2:7.3f} km2"
                if glacier_km2 is not None
                else ""
            )
        )
    return areas, glacier_areas


# -------------------------------------------------------------------- report
def report(areas=None, outline=None, glacier=None):
    if areas is None:
        areas = {}
        for m in MONTHS:
            p = MASK_DIR / f"{YEAR}_{m:02d}_snowice.png"
            if p.exists():
                areas[m] = cv2.imread(str(p), 0) > 0
    if not areas:
        raise SystemExit("no masks found, run mask first")
    if outline is None:
        outline = cv2.imread(str(OUTLINE_PNG), 0) > 0 if OUTLINE_PNG.exists() else None
    if glacier is None:
        glacier = {}
        for m in sorted(areas):
            p = MASK_DIR / f"{YEAR}_{m:02d}_glacier.png"
            if p.exists():
                glacier[m] = cv2.imread(str(p), 0) > 0

    counts = np.zeros_like(next(iter(areas.values())), dtype=np.uint8)
    rows = []
    for month in sorted(areas):
        mask = areas[month]
        counts += mask
        row = {
            "month": f"{YEAR}-{month:02d}",
            "name": calendar.month_name[month],
            "snow_ice_km2": round(int(mask.sum()) * PIXEL_M2 / 1e6, 3),
        }
        if month in glacier:
            row["glacier_km2"] = round(int(glacier[month].sum()) * PIXEL_M2 / 1e6, 3)
        rows.append(row)

    persistent = counts >= PERSIST_MIN_MONTHS
    persistent_km2 = int(persistent.sum()) * PIXEL_M2 / 1e6
    outline_km2 = int(outline.sum()) * PIXEL_M2 / 1e6 if outline is not None else None
    has_glacier = "glacier_km2" in rows[0]

    with open(CSV_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cv2.imwrite(
        str(MASK_DIR / "persistent_snowice_mask.png"),
        persistent.astype(np.uint8) * 255,
    )

    header = "month            snow+ice (km2)"
    if has_glacier:
        header += "   glacier (km2)"
    print("\n" + header)
    for r in rows:
        line = f"{r['name']:<16} {r['snow_ice_km2']:>13.3f}"
        if has_glacier:
            line += f"   {r['glacier_km2']:>13.3f}"
        print(line)
    s = [r["snow_ice_km2"] for r in rows]
    print(
        f"\nsnow+ice min/max/mean: {min(s):.3f} / {max(s):.3f} / {np.mean(s):.3f} km2"
    )
    if has_glacier:
        gm = [r["glacier_km2"] for r in rows]
        print(
            f"glacier  min/max/mean: {min(gm):.3f} / {max(gm):.3f} / {np.mean(gm):.3f} km2"
            f"  (prompted SAM2, per month)"
        )
    if outline_km2:
        print(f"reference outline (September): {outline_km2:.3f} km2")
    print(
        f"persistent snow+ice (>= {PERSIST_MIN_MONTHS} mo): {persistent_km2:.3f} km2"
    )
    print(f"wrote {CSV_PATH.name}, {PLOT_PATH.name}, "
          f"glacier_masks/persistent_snowice_mask.png")

    _plot(rows, outline_km2, persistent_km2, has_glacier)
    return rows, {"outline_km2": outline_km2, "persistent_km2": persistent_km2}


def _plot(rows, outline_km2, persistent_km2, has_glacier):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(
        x, [r["snow_ice_km2"] for r in rows], 0.65,
        label="monthly snow+ice extent (NDSI)", color="#1f77b4",
    )
    if has_glacier:
        ax.plot(
            x, [r["glacier_km2"] for r in rows], "o-", color="crimson", lw=1.8,
            label="glacier (prompted SAM2, re-run per month)",
        )
    if outline_km2:
        ax.axhline(
            outline_km2, color="darkred", ls="--", lw=1.2, alpha=0.7,
            label=f"September reference outline ({outline_km2:.2f} km$^2$)",
        )
    if persistent_km2:
        ax.axhline(
            persistent_km2, color="#ffbf00", ls=":", lw=1.5,
            label=f"persistent snow+ice (>={PERSIST_MIN_MONTHS} mo)",
        )
    ax.set_xticks(x, [r["name"][:3] for r in rows])
    ax.set_ylabel("classified area (km$^2$)")
    ax.set_title(
        f"Snow/ice and glacier surface area, {YEAR} — 15x15 km AOI, UTM 43N, "
        f"10 m (1 px = 100 m$^2$)"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------- cli
def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "stage",
        choices=["fetch", "outline", "glacier", "mask", "report", "all"],
        nargs="?",
        default="all",
    )
    parser.add_argument("--months", type=int, nargs="*", help="restrict to these months (1-12)")
    parser.add_argument("--force", action="store_true", help="re-fetch / rebuild outline")
    parser.add_argument("--device", choices=["mps", "cpu"], default=None)
    args = parser.parse_args(argv)
    months = args.months or MONTHS

    if args.stage in ("fetch", "all"):
        fetch(months, force=args.force)
    if args.stage in ("outline", "all"):
        outline_stage(device=args.device, force=args.force)
    if args.stage in ("glacier", "all"):
        glacier_monthly_stage(months, device=args.device)
    if args.stage in ("mask", "all"):
        mask_stage(months)
    if args.stage in ("report", "all"):
        report()


if __name__ == "__main__":
    sys.exit(main())
