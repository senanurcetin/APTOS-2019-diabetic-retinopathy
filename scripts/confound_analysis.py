"""Shortcut features and label noise.

A model scoring well does not mean it scores well for the right reason. This
script answers two questions:

1. SHORTCUT: how well can a model do without looking at the retina? A
   classifier trained only on file metadata (resolution, aspect ratio,
   brightness, file size) sets a floor. If that floor is high, part of the real
   model's score may come from the capture device rather than from pathology.

2. LABEL NOISE: where the same image appears twice with different labels, the
   disagreement rate bounds inter-rater agreement from below and therefore
   caps how well any model can do.

Output: reports/confounds_and_noise.md

Usage:
    python scripts/confound_analysis.py
"""
import pathlib

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import cross_val_predict

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATS = ROOT / "data" / "bq" / "image_stats.csv"
LABELS = ROOT / "data" / "bq" / "aptos_labels.csv"
PROBLEMS = ROOT / "reports" / "problem_images.csv"
REPORTS = ROOT / "reports"

GRADES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
META_COLS = ["width", "height", "aspect_ratio", "megapixels",
             "brightness", "contrast_std", "file_kb"]


def shortcut_baseline(m):
    """Classifier trained on metadata only - the shortcut floor."""
    X, y = m[META_COLS].values, m.diagnosis.values
    clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    pred = cross_val_predict(clf, X, y, cv=5, n_jobs=-1)

    clf.fit(X, y)
    importance = sorted(zip(META_COLS, clf.feature_importances_, strict=False),
                        key=lambda t: -t[1])
    return {
        "accuracy": float((pred == y).mean()),
        "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "all_zero_acc": float((y == 0).mean()),
        "importance": importance,
    }


def label_noise(problems, labels):
    """Estimate label noise from disagreements between duplicate pairs."""
    d = problems[problems.issue == "duplicate"].merge(labels, on="id_code",
                                                      suffixes=("", "_l"))
    pairs, conflicts, gaps = 0, 0, []
    for _, g in d.groupby("value"):
        v = g.diagnosis.values
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                pairs += 1
                if v[i] != v[j]:
                    conflicts += 1
                    gaps.append(abs(int(v[i]) - int(v[j])))
    if pairs == 0:
        return None

    agree = 1 - conflicts / pairs
    return {
        "groups": d.value.nunique(),
        "pairs": pairs,
        "conflicts": conflicts,
        "rate": conflicts / pairs,
        "agree": agree,
        # If two independent labels are each correct with probability p, the
        # agreement rate is roughly p^2; sqrt(agreement) is a rough inversion.
        "single_label_acc": float(np.sqrt(agree)),
        "gap_dist": dict(pd.Series(gaps).value_counts().sort_index()) if gaps else {},
        "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
    }


def build(m, sc, ln):
    md = ["# Shortcut Features and Label Noise\n\n"]
    md.append("Two analyses asking whether the model earns its score for the "
              "right reason, and how high that score could possibly go.\n")

    # ---------------------------------------------------------- shortcut
    md.append("\n## 1. The metadata shortcut\n\n")
    md.append("A RandomForest trained only on file properties - resolution, "
              "aspect ratio, brightness, contrast, file size. **It never looks "
              "at the retina.** Five-fold cross-validation.\n\n")
    md.append("| measure | metadata floor | always-predict-0 |\n|---|---|---|\n")
    md.append(f"| Accuracy | {sc['accuracy']:.4f} | {sc['all_zero_acc']:.4f} |\n")
    md.append(f"| QWK | **{sc['qwk']:.4f}** | 0.0000 |\n")

    md.append("\nMost informative metadata features:\n\n")
    md.append("| feature | importance |\n|---|---|\n")
    for name, imp in sc["importance"][:5]:
        md.append(f"| `{name}` | {imp:.3f} |\n")

    sq = m[(m.width == 1050) & (m.height == 1050)]
    rest = m[~((m.width == 1050) & (m.height == 1050))]
    md.append(f"\nWhere it comes from: **{(sq.diagnosis == 0).mean() * 100:.1f}%** of "
              f"the 1050x1050 images (the most common resolution) are `No DR`, "
              f"against **{(rest.diagnosis == 0).mean() * 100:.1f}%** at every other "
              "resolution.\n\n")
    md.append(f"| grade | 1050x1050 (n={len(sq)}) | other (n={len(rest)}) |\n|---|---|---|\n")
    for g in range(5):
        a, b = (sq.diagnosis == g).sum(), (rest.diagnosis == g).sum()
        md.append(f"| {g} {GRADES[g]} | {a} ({a / len(sq) * 100:.1f}%) | "
                  f"{b} ({b / len(rest) * 100:.1f}%) |\n")

    md.append("\n### What this means\n\n")
    md.append("APTOS data was collected across several sites in India with "
              "different cameras. Resolution is the device's signature, and "
              "device correlates strongly with disease prevalence. Even though "
              "the model sees resized images, aspect ratio, sharpness and edge "
              "geometry still carry that information.\n\n")
    md.append(f"This does not invalidate the results, but it does mean **QWK "
              f"{sc['qwk']:.3f} is reachable without any diagnosis at all**. "
              "Reported scores should be read alongside this floor.\n")

    # ------------------------------------------------------ class properties
    md.append("\n## 2. Image properties by class\n\n")
    md.append("| grade | n | brightness | contrast | megapixels | square |\n"
              "|---|---|---|---|---|---|\n")
    for g in range(5):
        s = m[m.diagnosis == g]
        square = ((s.aspect_ratio > 0.98) & (s.aspect_ratio < 1.02)).mean() * 100
        md.append(f"| {g} {GRADES[g]} | {len(s)} | {s.brightness.median():.1f} | "
                  f"{s.contrast_std.median():.1f} | {s.megapixels.median():.2f} | "
                  f"{square:.0f}% |\n")

    md.append("\nKruskal-Wallis test for differences between classes:\n\n")
    md.append("| feature | p | result |\n|---|---|---|\n")
    for col in ["brightness", "contrast_std", "megapixels"]:
        _, p = st.kruskal(*[m[m.diagnosis == g][col].values for g in range(5)])
        md.append(f"| {col} | {p:.2e} | {'differs' if p < 0.05 else 'no difference'} |\n")

    # Computed, not written in. This sentence used to be literal text inside a
    # generated report, so it would have kept describing the data after the data
    # changed.
    def square_share(frame):
        return float(((frame.aspect_ratio > 0.98) & (frame.aspect_ratio < 1.02)).mean())

    no_dr = m[m.diagnosis == 0]
    diseased = [m[m.diagnosis == g] for g in range(1, 5)]
    sick_mp = [s.megapixels.median() for s in diseased if len(s)]
    sick_sq = [square_share(s) for s in diseased if len(s)]
    md.append(
        f"\n`No DR` images have a median of {no_dr.megapixels.median():.2f} megapixels "
        f"and {square_share(no_dr):.0%} are square; the diseased grades have medians "
        f"of {min(sick_mp):.1f}-{max(sick_mp):.1f} megapixels, and at most "
        f"{max(sick_sq):.0%} of any diseased grade is square."
    )
    if no_dr.megapixels.median() * 2 < min(sick_mp):
        md.append(" That is the shortcut, stated directly.")
    md.append("\n")

    # --------------------------------------------------------- label noise
    md.append("\n## 3. Label noise\n\n")
    if ln is None:
        md.append("No duplicate pairs found; no estimate possible.\n")
    else:
        md.append("Where the same image appears more than once in the dataset, "
                  "its labels should match. The cases where they do not give a "
                  "lower bound on inter-rater agreement.\n\n")
        md.append("| measure | value |\n|---|---|\n")
        md.append(f"| Verified duplicate groups | {ln['groups']} |\n")
        md.append(f"| Same-image pairs | {ln['pairs']} |\n")
        md.append(f"| Pairs with conflicting labels | {ln['conflicts']} "
                  f"({ln['rate'] * 100:.1f}%) |\n")
        md.append(f"| Mean disagreement size | {ln['mean_gap']:.2f} grades |\n")
        md.append(f"| Agreement rate | {ln['agree'] * 100:.1f}% |\n")
        md.append(f"| Estimated single-label accuracy | "
                  f"{ln['single_label_acc'] * 100:.1f}% |\n")
        if ln["gap_dist"]:
            dist = ", ".join(f"{k} grade(s): {v}" for k, v in ln["gap_dist"].items())
            md.append(f"\nDisagreement sizes: {dist}.\n")
        md.append("\n### What this means\n\n")
        # This report runs before any training, so it says nothing about a
        # particular model. It used to carry the literal sentence "The current
        # model sits near 0.82 on test", which was a claim this stage could not
        # know and which went stale when the model changed. The comparison lives
        # in RESULTS.md, next to the model numbers.
        md.append(f"A single label is correct roughly "
                  f"**{ln['single_label_acc'] * 100:.0f}%** of the time. That caps "
                  "the accuracy any model, however good, can reach on this "
                  "dataset: a model near that figure is close to the ceiling, "
                  "and part of its remaining error belongs to the labels rather "
                  "than to the model. RESULTS.md reads the trained models "
                  "against this number.\n\n")
        md.append("This is a **lower bound**: it only measures noise visible in "
                  "duplicated images, not in the rest of the dataset.\n")

    md.append("\n---\n\nGenerated by `python scripts/confound_analysis.py`\n")
    return "".join(md)


def main():
    for f in (STATS, LABELS, PROBLEMS):
        if not f.exists():
            raise SystemExit(f"{f} not found - run scan_images.py and "
                             "quality_report.py first")

    stats = pd.read_csv(STATS)
    stats = stats[stats.readable]
    labels = pd.read_csv(LABELS)[["id_code", "diagnosis", "split"]]
    problems = pd.read_csv(PROBLEMS)

    m = stats.merge(labels, on="id_code", suffixes=("", "_l"))

    print("computing the metadata shortcut floor (5-fold CV)...")
    sc = shortcut_baseline(m)
    print(f"  QWK={sc['qwk']:.4f}  accuracy={sc['accuracy']:.4f}")

    print("estimating label noise...")
    ln = label_noise(problems, labels)
    if ln:
        print(f"  conflicting pairs: {ln['conflicts']}/{ln['pairs']} "
              f"({ln['rate'] * 100:.1f}%)")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "confounds_and_noise.md").write_text(build(m, sc, ln), encoding="utf-8")
    print(f"\n-> {REPORTS / 'confounds_and_noise.md'}")


if __name__ == "__main__":
    main()
