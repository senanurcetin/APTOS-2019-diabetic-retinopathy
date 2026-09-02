"""Scan raw retina images for their properties and write them to BigQuery.

One pass extracts everything: dimensions, resolution, aspect ratio, colour mode,
channel count, brightness/contrast statistics and a perceptual hash. The quality
report (quality_report.py) and the confound analysis derive from this table, so
the 8 GB of source images are read only once.

Output: BigQuery `aptos_image_stats` + data/bq/image_stats.csv

Usage:
    python scripts/scan_images.py
    python scripts/scan_images.py --no-bq
"""
import argparse
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from preprocessing import auto_crop, dhash  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

SOURCE_DIRS = {
    "train": ROOT / "data" / "images" / "train_images" / "train_images",
    "valid": ROOT / "data" / "images" / "val_images" / "val_images",
    "test": ROOT / "data" / "images" / "test_images" / "test_images",
}

# Cloud identifiers are read from the environment so the project runs against
# any BigQuery project and bucket. The defaults are the ones this work used.
PROJECT_ID = os.environ.get("APTOS_GCP_PROJECT", "datascientis")
BQ_DATASET = os.environ.get("APTOS_BQ_DATASET", "APTOS_2019")


def scan_one(args):
    path, split = args
    row = {"id_code": path.stem, "split": split, "readable": False}

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        row["error"] = "unreadable"
        return row

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Colour mode is judged by content, not by file format: an image whose three
    # channels are identical is grayscale even if stored as RGB.
    b, g, r = cv2.split(img)
    is_gray = bool((b == g).all() and (g == r).all())

    cropped = auto_crop(img)
    ch, cw = cropped.shape[:2]

    row.update({
        "readable": True,
        "error": None,
        "width": int(w),
        "height": int(h),
        "megapixels": round(w * h / 1e6, 3),
        "aspect_ratio": round(w / h, 4),
        "channels": int(img.shape[2]),
        "color_mode": "Grayscale" if is_gray else "RGB",
        "file_kb": round(path.stat().st_size / 1024, 1),
        "brightness": round(float(gray.mean()), 2),
        "contrast_std": round(float(gray.std()), 2),
        "black_ratio": round(float((gray <= 7).mean()), 4),
        "crop_width": int(cw),
        "crop_height": int(ch),
        "crop_saving": round(1 - (cw * ch) / (w * h), 4),
        "dhash": dhash(img),
    })
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-bq", action="store_true")
    args = ap.parse_args()

    jobs = []
    for split, d in SOURCE_DIRS.items():
        if not d.exists():
            raise SystemExit(f"{d} not found - download the Kaggle data first")
        jobs += [(p, split) for p in sorted(d.glob("*.png"))]

    print(f"scanning {len(jobs)} images with {args.workers} workers...")

    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(scan_one, j) for j in jobs]):
            rows.append(fut.result())
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)}")

    df = pd.DataFrame(rows).sort_values(["split", "id_code"]).reset_index(drop=True)

    out = ROOT / "data" / "bq" / "image_stats.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n{len(df)} rows -> {out}")

    ok = df[df.readable]
    print(f"\nreadable    : {len(ok)}/{len(df)}")
    print(f"colour mode : {ok.color_mode.value_counts().to_dict()}")
    print(f"channels    : {ok.channels.value_counts().to_dict()}")
    print("\nresolution:")
    print(f"  width      {ok.width.min()}-{ok.width.max()}  (median {int(ok.width.median())})")
    print(f"  height     {ok.height.min()}-{ok.height.max()}  (median {int(ok.height.median())})")
    print(f"  megapixels {ok.megapixels.min():.2f}-{ok.megapixels.max():.2f}")
    print(f"  aspect     {ok.aspect_ratio.min():.3f}-{ok.aspect_ratio.max():.3f}")
    print(f"  distinct resolutions: {ok.groupby(['width','height']).ngroups}")
    print("\nfive most common resolutions:")
    for (w, h), n in ok.groupby(["width", "height"]).size().nlargest(5).items():
        print(f"  {w}x{h}  {n}")
    print(f"\nbrightness: {ok.brightness.min():.1f}-{ok.brightness.max():.1f} "
          f"(mean {ok.brightness.mean():.1f})")
    print(f"contrast  : {ok.contrast_std.min():.1f}-{ok.contrast_std.max():.1f} "
          f"(mean {ok.contrast_std.mean():.1f})")
    print(f"area removed by auto-crop: {ok.crop_saving.mean() * 100:.1f}% on average")

    if args.no_bq:
        return

    from google.cloud import bigquery
    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{BQ_DATASET}.aptos_image_stats"
    cfg = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        clustering_fields=["split"],
    )
    client.load_table_from_dataframe(df, table_id, job_config=cfg).result()
    print(f"\n{table_id} <- {len(df)} rows")


if __name__ == "__main__":
    main()
