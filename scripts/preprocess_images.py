"""Prepare APTOS images for training.

Raw PNGs average 2 MB and are mostly black space outside the retina. Decoding
them every epoch would make training I/O-bound. This script runs once and
applies the shared pipeline from scripts/preprocessing.py to every image:

    Read -> Quality check -> Auto-crop -> CLAHE -> Square -> Resize

Normalisation is deliberately NOT done here; it happens in the torchvision
transforms at training time (train.py). Keeping it in one place avoids
normalising twice.

Input : data/images/{train_images/train_images, val_images/val_images,
                    test_images/test_images}/*.png
Output: data/processed_clahe/<split>/<id_code>.jpg   (default)
        data/processed/<split>/<id_code>.jpg         (with --no-clahe)

Each output directory also gets a _manifest.json recording the settings it was
produced with; train.py reads and prints it, so every run records which
preprocessing it used.

Usage:
    python scripts/preprocess_images.py --size 512               # with CLAHE
    python scripts/preprocess_images.py --size 512 --no-clahe    # control set
"""
import argparse
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from preprocessing import preprocess  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The Kaggle archive is inconsistent: the validation split lives under
# "val_images", not "valid_images".
SOURCE_DIRS = {
    "train": ROOT / "data" / "images" / "train_images" / "train_images",
    "valid": ROOT / "data" / "images" / "val_images" / "val_images",
    "test": ROOT / "data" / "images" / "test_images" / "test_images",
}


def process_one(args):
    src, dst, size, use_clahe, clip, square_mode = args
    img, info = preprocess(src, size=size, use_clahe=use_clahe,
                           normalize=False, clip_limit=clip,
                           quality_check=True, square_mode=square_mode)
    if img is None:
        return src.stem, info.get("error", "unknown error")

    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return src.stem, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--clip-limit", type=float, default=2.0)
    ap.add_argument("--square-mode", choices=["squash", "pad"], default="squash",
                    help="how to reach a square; squash cuts black area from "
                         "28.5%% to 13.4%% without losing tissue")
    ap.add_argument("--no-clahe", action="store_true",
                    help="produce without CLAHE - the control set for comparison")
    ap.add_argument("--out", default=None,
                    help="output directory; defaults by CLAHE setting")
    args = ap.parse_args()

    use_clahe = not args.no_clahe
    out_root = (ROOT / args.out if args.out else
                ROOT / "data" / ("processed_clahe" if use_clahe else "processed"))

    jobs = []
    for split, src_dir in SOURCE_DIRS.items():
        if not src_dir.exists():
            raise SystemExit(f"{src_dir} not found - download the Kaggle data first")
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for png in sorted(src_dir.glob("*.png")):
            jobs.append((png, out_dir / f"{png.stem}.jpg", args.size,
                         use_clahe, args.clip_limit, args.square_mode))

    print(f"{len(jobs)} images, {args.size}x{args.size}, "
          f"CLAHE {'on (clip=' + str(args.clip_limit) + ')' if use_clahe else 'OFF'}, "
          f"square={args.square_mode}, {args.workers} workers")
    print(f"output: {out_root}")

    failures, done = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(process_one, j) for j in jobs]):
            stem, err = fut.result()
            done += 1
            if err:
                failures.append((stem, err))
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)}")

    # Provenance record: which settings produced this directory. train.py and
    # aptos.training checks it, so a run always states its preprocessing.
    (out_root / "_manifest.json").write_text(json.dumps({
        "size": args.size,
        "clahe": use_clahe,
        "clip_limit": args.clip_limit if use_clahe else None,
        "square_mode": args.square_mode,
        "n_images": done - len(failures),
    }, indent=2), encoding="utf-8")

    print(f"\ndone: {done - len(failures)} written, {len(failures)} skipped")
    print(f"manifest: {out_root / '_manifest.json'}")
    for stem, err in failures[:20]:
        print(f"  {stem}: {err}")
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more")


if __name__ == "__main__":
    main()
