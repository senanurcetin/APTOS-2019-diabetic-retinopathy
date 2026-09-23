"""Data quality report and problem-image list.

Derives from the table produced by scan_images.py; source images are not read
again except to verify duplicate candidates. Checks:

  - unreadable / corrupt images
  - labels without an image, images without a label
  - unusably dark or bright images, plus distribution outliers
  - duplicate images (perceptual hash, then pixel-level verification)
  - cross-split duplicates, which would inflate the reported score

Output: reports/data_quality.md
        reports/problem_images.csv
        reports/leaked_train_ids.csv   (training images to exclude)

Usage:
    python scripts/quality_report.py
"""
import argparse
import pathlib
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from preprocessing import HARD_BRIGHT, HARD_DARK, brightness_outliers  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATS = ROOT / "data" / "bq" / "image_stats.csv"
LABELS = ROOT / "data" / "bq" / "aptos_labels.csv"
REPORTS = ROOT / "reports"

PROCESSED = ROOT / "data" / "processed"


def section(title):
    return f"\n## {title}\n\n"


# Module-level so it actually persists. Duplicate verification compares each
# candidate against several others, so the same thumbnail is asked for many
# times; decoding it once is the whole point of this cache.
_THUMB_CACHE: dict = {}


def _thumb(id_code, split):
    """128px grayscale thumbnail from the processed JPEG - far cheaper than
    re-reading the raw PNG."""
    cache = _THUMB_CACHE
    key = (id_code, split)
    if key in cache:
        return cache[key]

    path = PROCESSED / split / f"{id_code}.jpg"
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA).astype("float32")
    cache[key] = img
    return img


def _same_image(a, b, min_corr=0.995, max_mae=3.0):
    """Whether two thumbnails are genuinely the same image.

    The thresholds are deliberately strict. When this was measured, true pairs
    scored correlation 1.0000 / MAE 0.00 while different retinas sat at
    0.97-0.985; the cutoff falls in the gap between them.
    """
    if a is None or b is None:
        return False
    return (float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) >= min_corr
            and float(np.abs(a - b).mean()) <= max_mae)


def _verify_duplicates(candidates):
    """Verify dHash candidates at pixel level and return the real groups."""
    groups = []
    for _, g in candidates.groupby("dhash"):
        members = list(zip(g.id_code, g.split, strict=False))
        remaining = list(members)
        while len(remaining) > 1:
            head, rest = remaining[0], remaining[1:]
            matched, leftover = [head], []
            for other in rest:
                if _same_image(_thumb(*head), _thumb(*other)):
                    matched.append(other)
                else:
                    leftover.append(other)
            if len(matched) > 1:
                groups.append(matched)
            remaining = leftover
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dark", type=float, default=HARD_DARK,
                    help="below this an image counts as unusable")
    ap.add_argument("--bright", type=float, default=HARD_BRIGHT,
                    help="above this an image counts as unusable")
    ap.add_argument("--mad-k", type=float, default=3.5,
                    help="MAD multiplier for brightness outliers")
    args = ap.parse_args()

    if not STATS.exists():
        raise SystemExit(f"{STATS} not found - run scripts/scan_images.py first")

    stats = pd.read_csv(STATS)
    labels = pd.read_csv(LABELS)
    REPORTS.mkdir(exist_ok=True)

    md = ["# Data Quality Report\n",
          f"\nSource: {len(stats)} scanned images, {len(labels)} label rows.\n"]
    problems = []

    # ------------------------------------------------------------ readability
    unreadable = stats[~stats.readable]
    md.append(section("Corrupt / unreadable images"))
    md.append(f"- Scanned: **{len(stats)}**\n")
    md.append(f"- Unreadable: **{len(unreadable)}**\n")
    for _, r in unreadable.iterrows():
        problems.append({"id_code": r.id_code, "split": r.split,
                         "issue": "unreadable", "value": r.get("error")})

    ok = stats[stats.readable].copy()

    # ------------------------------------------------------ label <-> file
    md.append(section("Label and file correspondence"))
    lab_ids, img_ids = set(labels.id_code), set(ok.id_code)
    only_lab, only_img = lab_ids - img_ids, img_ids - lab_ids
    md.append(f"- Labelled but no image: **{len(only_lab)}**\n")
    md.append(f"- Image but no label: **{len(only_img)}**\n")
    for i in sorted(only_lab):
        problems.append({"id_code": i, "split": "", "issue": "no image", "value": ""})
    for i in sorted(only_img):
        problems.append({"id_code": i, "split": "", "issue": "no label", "value": ""})

    if PROCESSED.exists():
        proc = {p.stem for split in ("train", "valid", "test")
                for p in (PROCESSED / split).glob("*.jpg")}
        md.append(f"- Missing from the processed set: **{len(lab_ids - proc)}**\n")

    # ------------------------------------------------------------- brightness
    # Two separate questions, two separate measures:
    #   1) Is the image usable at all?   -> fixed thresholds (near black/white)
    #   2) Does it deviate from the rest? -> MAD-based outlier detection
    # A fixed threshold alone is misleading here: fundus photographs are
    # inherently dark and brightness never exceeds 130 in this dataset, so a
    # general-purpose "too bright" cutoff can never fire and its zero means
    # nothing.
    dark = ok[ok.brightness < args.dark]
    bright = ok[ok.brightness > args.bright]
    outliers = ok[brightness_outliers(ok.brightness.values, k=args.mad_k)]

    md.append(section("Brightness checks"))
    md.append("Two measures are used: fixed thresholds for usability, and a "
              "distribution-based test for unusualness.\n\n")
    md.append("| measure | threshold | flagged |\n|---|---|---|\n")
    md.append(f"| Unusably dark | `< {args.dark}` | {len(dark)} |\n")
    md.append(f"| Unusably bright | `> {args.bright}` | {len(bright)} |\n")
    md.append(f"| Distribution outlier | MAD, k={args.mad_k} | {len(outliers)} |\n")
    md.append(f"\n- Brightness range: {ok.brightness.min():.1f} - "
              f"{ok.brightness.max():.1f} (median {ok.brightness.median():.1f})\n")
    md.append(f"- Contrast (std) range: {ok.contrast_std.min():.1f} - "
              f"{ok.contrast_std.max():.1f} (median {ok.contrast_std.median():.1f})\n")

    if len(outliers) == 0:
        md.append("\nNo MAD outliers. This is an informative zero: the measure "
                  "adapts to the data's own scale, so it means nothing in the "
                  "brightness distribution genuinely deviates.\n")
    else:
        md.append("\n| id_code | split | brightness |\n|---|---|---|\n")
        for _, r in outliers.iterrows():
            md.append(f"| {r.id_code} | {r.split} | {r.brightness:.1f} |\n")

    for frame, issue in ((dark, "too dark"), (bright, "too bright"),
                         (outliers, "brightness outlier")):
        for _, r in frame.iterrows():
            problems.append({"id_code": r.id_code, "split": r.split,
                             "issue": issue, "value": r.brightness})

    # -------------------------------------------------------------- duplicates
    md.append(section("Duplicate images"))
    candidates = ok.groupby("dhash").filter(lambda g: len(g) > 1)
    n_cand = candidates.dhash.nunique() if len(candidates) else 0

    md.append("Two-stage detection: dHash is a cheap candidate generator, then "
              "every candidate pair is verified at pixel level.\n\n")
    md.append(f"- dHash candidate groups: **{n_cand}**\n")

    # dHash alone is not enough: every fundus image is a bright disc on black,
    # so different retinas can produce the same hash. Candidates are verified
    # with correlation and mean absolute error over 128px thumbnails.
    confirmed, cross = [], 0
    if n_cand:
        for members in _verify_duplicates(candidates):
            splits = sorted({m[1] for m in members})
            if len(splits) > 1:
                cross += 1
            confirmed.append((members, splits))
            for id_code, split in members:
                problems.append({"id_code": id_code, "split": split,
                                 "issue": "duplicate",
                                 "value": "+".join(m[0] for m in members)})

        md.append(f"- Verified at pixel level: **{len(confirmed)}**\n")
        md.append(f"- False positives removed: **{n_cand - len(confirmed)}**\n")
        md.append(f"- Images affected: **{sum(len(m) for m, _ in confirmed)}**\n")
        md.append(f"- Spanning more than one split: **{cross}**\n")

        if confirmed:
            md.append("\n| images | splits |\n|---|---|\n")
            for members, splits in confirmed:
                md.append(f"| {', '.join(m[0] for m in members)} | "
                          f"{', '.join(splits)} |\n")
        if cross:
            md.append("\n> Cross-split duplicates inflate the reported score: if "
                      "the same image appears in both training and test, the test "
                      "result is optimistic. These should be dropped from training.\n")
    else:
        md.append("\nNo candidates found.\n")
    n_groups = len(confirmed)

    # -------------------------------------------------------------- summary
    md.insert(2, section("Summary"))
    md.insert(3,
              f"| check | result |\n|---|---|\n"
              f"| Unreadable images | {len(unreadable)} |\n"
              f"| Images without a label | {len(only_img)} |\n"
              f"| Labels without an image | {len(only_lab)} |\n"
              f"| Unusably dark | {len(dark)} |\n"
              f"| Unusably bright | {len(bright)} |\n"
              f"| Brightness outliers (MAD k={args.mad_k}) | {len(outliers)} |\n"
              f"| Duplicate groups (verified) | {n_groups} |\n"
              f"| Cross-split duplicates | {cross} |\n")

    (REPORTS / "data_quality.md").write_text("".join(md), encoding="utf-8")

    prob_df = pd.DataFrame(problems, columns=["id_code", "split", "issue", "value"])
    prob_df.to_csv(REPORTS / "problem_images.csv", index=False)

    # Training images to exclude: those with an exact copy in valid or test.
    # Dropping the training side keeps the evaluation sets intact, so runs stay
    # comparable with each other.
    leaked = sorted({id_code for members, splits in confirmed
                     if len(splits) > 1
                     for id_code, split in members if split == "train"})
    pd.DataFrame({"id_code": leaked}).to_csv(REPORTS / "leaked_train_ids.csv",
                                             index=False)

    print("".join(md))
    print(f"\n-> {REPORTS / 'data_quality.md'}")
    print(f"-> {REPORTS / 'problem_images.csv'}  ({len(prob_df)} rows)")
    print(f"-> {REPORTS / 'leaked_train_ids.csv'}  ({len(leaked)} training images)")


if __name__ == "__main__":
    main()
