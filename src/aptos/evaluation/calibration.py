"""Calibration and the clinical operating point.

Two things this project reported but never examined.

**The operating point.** Grade QWK is a benchmark metric. A screening programme
asks a different question - refer this person, or not - and picks a point on the
sensitivity/specificity curve deliberately, because the two errors cost wildly
different amounts. Missing a referable case risks sight; a false positive costs
one appointment. Everything here selects at **sensitivity >= 0.90** and reports
what that costs in specificity and workload.

**Calibration.** IDRiD exposed this: the model issues 8 grade-4 predictions
where 64 exist, because thresholds fitted on APTOS validation are carried across
unchanged. Missed referrals stay at 2 of 148, so it is compressing the scale
rather than failing to detect - but "compressing the scale" is exactly what
calibration measures, and it had never been measured.

Selection happens on **out-of-fold** predictions, never on test. The fold
checkpoints make that possible after the fact: each fold model predicts the
slice it was held out from, so the pool gets a clean prediction from a model
that never saw it.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from aptos.config import REFERABLE_FROM, Config
from aptos.data import labels as labels_mod
from aptos.training.cv import build_folds


def out_of_fold_predictions(cfg: Config, sweep: pathlib.Path) -> pd.DataFrame:
    """Predict every pool image with the fold model that held it out.

    This is what makes an honest operating-point choice possible. Selecting a
    threshold on the test set and then reporting test performance at that
    threshold would be measuring the selection, not the model.
    """
    import torch
    from torch.utils.data import DataLoader

    from aptos.evaluation.external import FlatDataset
    from aptos.training.loop import build_model, build_transforms, pick_device

    device = pick_device()
    df = labels_mod.load_labels(cfg)
    if cfg.train.exclude_leaked:
        df, _ = labels_mod.exclude_leaked(df, cfg)
    pool, _ = labels_mod.pool_and_holdout(df, cfg)

    splits = build_folds(cfg, pool)
    checkpoints = sorted(sweep.glob("fold*.pt"), key=lambda p: int(p.stem[4:]))
    if len(checkpoints) != len(splits):
        raise RuntimeError(
            f"{len(checkpoints)} checkpoints against {len(splits)} folds - the sweep "
            f"is incomplete, or its config differs from this one"
        )

    _, eval_tf = build_transforms(cfg.train.size, cfg.augment)
    model = build_model(cfg, device)
    model.eval()

    raw = np.zeros(len(pool), dtype=float)
    covered = np.zeros(len(pool), dtype=bool)

    for (_, va_idx), ckpt in zip(splits, checkpoints, strict=True):
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state_dict"])

        slice_df = pool.iloc[va_idx]
        # Fold datasets read from the ORIGINAL split directory, since an image's
        # role changes between folds but its location on disk does not.
        frames = []
        for orig_split, group in slice_df.groupby("orig_split"):
            loader = DataLoader(
                FlatDataset(group, cfg.data_dir / orig_split, eval_tf),
                batch_size=cfg.train.batch, shuffle=False,
                num_workers=cfg.train.workers, pin_memory=True,
            )
            outputs = []
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(device, non_blocking=True)
                    with torch.autocast("cuda", enabled=device == "cuda"):
                        outputs.append(model(images).squeeze(1).float().cpu().numpy())
            frames.append(pd.Series(np.concatenate(outputs), index=group.index))
        predicted = pd.concat(frames).reindex(slice_df.index)
        raw[va_idx] = predicted.to_numpy()
        covered[va_idx] = True
        print(f"  fold {ckpt.stem[4:]}: {len(va_idx)} out-of-fold predictions")

    if not covered.all():
        raise RuntimeError(f"{(~covered).sum()} pool images received no prediction")

    out = pool[["id_code", "diagnosis", "orig_split"]].copy()
    out["raw"] = raw
    return out


# ------------------------------------------------------------- operating point

def pick_operating_point(y_referable, score, min_sensitivity: float = 0.90) -> dict:
    """The lowest-cost cut that still reaches the required sensitivity.

    Sensitivity is the constraint, not the objective: in screening a missed
    referral risks sight while a false positive costs an appointment. Among the
    cuts that clear the bar, this takes the one with the best specificity.
    """
    y = np.asarray(y_referable).astype(int)
    score = np.asarray(score, dtype=float)
    fpr, tpr, cuts = roc_curve(y, score)

    eligible = tpr >= min_sensitivity
    if not eligible.any():
        raise ValueError(f"no threshold reaches sensitivity {min_sensitivity}")
    # Among eligible points the first is the one with the lowest FPR, because
    # roc_curve returns thresholds in decreasing order.
    idx = int(np.argmax(eligible))
    return {
        "min_sensitivity": min_sensitivity,
        "cut": float(cuts[idx]),
        "sensitivity": float(tpr[idx]),
        "specificity": float(1 - fpr[idx]),
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
    }


def apply_operating_point(y_referable, score, cut: float) -> dict:
    y = np.asarray(y_referable).astype(int)
    flag = np.asarray(score, dtype=float) >= cut
    tp = int((y.astype(bool) & flag).sum())
    fn = int((y.astype(bool) & ~flag).sum())
    tn = int((~y.astype(bool) & ~flag).sum())
    fp = int((~y.astype(bool) & flag).sum())
    return {
        "cut": float(cut), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "ppv": tp / (tp + fp) if (tp + fp) else float("nan"),
        # What share of the population a clinician ends up reviewing.
        "referral_rate": float(flag.mean()),
        "roc_auc": float(roc_auc_score(y, score)) if len(set(y)) > 1 else float("nan"),
    }


# ------------------------------------------------------------------ calibration

def fit_calibrator(y_referable, score) -> LogisticRegression:
    """Platt scaling: turn the regression output into a referral probability.

    The model emits an ordinal severity score, not a probability, so there is
    nothing to temperature-scale. A one-dimensional logistic fit on out-of-fold
    predictions is the equivalent step, and it is what makes a reliability
    diagram or a decision curve meaningful.
    """
    model = LogisticRegression()
    model.fit(np.asarray(score, dtype=float).reshape(-1, 1),
              np.asarray(y_referable).astype(int))
    return model


def probabilities(calibrator: LogisticRegression, score) -> np.ndarray:
    return calibrator.predict_proba(np.asarray(score, dtype=float).reshape(-1, 1))[:, 1]


def expected_calibration_error(y, prob, bins: int = 10) -> tuple[float, pd.DataFrame]:
    """ECE and the reliability table it is computed from.

    The table is returned as well because a single ECE number hides direction:
    a model can be over-confident at the top and under-confident at the bottom
    and still average out.
    """
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    rows, error = [], 0.0

    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (prob > lo) & (prob <= hi) if lo > 0 else (prob >= lo) & (prob <= hi)
        if not mask.any():
            continue
        confidence, observed = float(prob[mask].mean()), float(y[mask].mean())
        weight = mask.sum() / len(prob)
        error += weight * abs(confidence - observed)
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()),
                     "mean_predicted": confidence, "observed": observed,
                     "gap": observed - confidence})
    return error, pd.DataFrame(rows)


def decision_curve(y, prob, thresholds=None) -> pd.DataFrame:
    """Net benefit against treat-all and treat-none.

    Net benefit = TP/n - FP/n * (pt / (1 - pt)), where pt is the probability
    threshold at which a clinician would act. It answers whether acting on the
    model beats the two trivial policies, across the range of preferences a
    clinician might hold, rather than at one arbitrary cut.
    """
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    n = len(y)
    prevalence = y.mean()
    thresholds = np.arange(0.05, 0.75, 0.05) if thresholds is None else np.asarray(thresholds)

    rows = []
    for pt in thresholds:
        flag = prob >= pt
        tp = int((flag & y.astype(bool)).sum())
        fp = int((flag & ~y.astype(bool)).sum())
        odds = pt / (1 - pt)
        rows.append({
            "threshold": float(pt),
            "model": tp / n - (fp / n) * odds,
            "treat_all": prevalence - (1 - prevalence) * odds,
            "treat_none": 0.0,
        })
    table = pd.DataFrame(rows)
    table["model_beats_both"] = (table["model"] > table["treat_all"]) & (table["model"] > 0)
    return table


# ------------------------------------------------------------ site transfer

def calibration_in_the_large(y, prob) -> float:
    """Observed rate minus mean predicted probability.

    Positive means under-confident - more referable cases than the model
    expects - which is the direction IDRiD showed.
    """
    return float(np.asarray(y, dtype=float).mean() - np.asarray(prob, dtype=float).mean())


def site_transfer(fit: pd.DataFrame, test: pd.DataFrame, aptos_calibrator,
                  aptos_cut: float, min_sensitivity: float = 0.90) -> dict:
    """Recalibrate on one new site, then measure the effect on another.

    The question a deployment faces: labels from a new site are scarce, so is
    one local recalibration enough for the next site too, or does every site
    need its own? `fit` plays the site that supplied labels; `test` is the one
    the result is judged on. Both frames need `referable` and `raw`.
    """
    local = fit_calibrator(fit["referable"], fit["raw"])
    local_point = pick_operating_point(fit["referable"], fit["raw"], min_sensitivity)
    before, _ = expected_calibration_error(test["referable"],
                                           probabilities(aptos_calibrator, test["raw"]))
    after, _ = expected_calibration_error(test["referable"],
                                          probabilities(local, test["raw"]))
    at_aptos = apply_operating_point(test["referable"], test["raw"], aptos_cut)
    at_local = apply_operating_point(test["referable"], test["raw"], local_point["cut"])
    return {
        "ece_aptos_calibrator": float(before),
        "ece_refitted": float(after),
        "cut_aptos": float(aptos_cut),
        "cut_refitted": float(local_point["cut"]),
        "sensitivity_aptos_cut": at_aptos["sensitivity"],
        "sensitivity_refitted_cut": at_local["sensitivity"],
        "specificity_aptos_cut": at_aptos["specificity"],
        "specificity_refitted_cut": at_local["specificity"],
    }


# --------------------------------------------------------------------- report

def _md_table(frame: pd.DataFrame, floats: int = 3) -> list[str]:
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    lines = [header, rule]
    for _, row in frame.iterrows():
        cells = [f"{v:.{floats}f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main(argv=None) -> int:
    import argparse

    from aptos.evaluation import external
    from aptos.evaluation.confound import load_sweep_predictions

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--min-sensitivity", type=float, default=0.90)
    parser.add_argument("--out", default="reports/calibration.md")
    args = parser.parse_args(argv)

    sweep = pathlib.Path(args.sweep)
    variant = sweep.name.split("-")[0]
    cfg = Config.load(f"configs/{variant}.yaml")
    base = Config.load("configs/cv.yaml")
    cfg.train, cfg.cv, cfg.tracking = base.train, base.cv, base.tracking

    print("out-of-fold predictions on the pool (selection set):")
    oof = out_of_fold_predictions(cfg, sweep)
    oof["referable"] = oof["diagnosis"] >= REFERABLE_FROM

    point = pick_operating_point(oof["referable"], oof["raw"], args.min_sensitivity)
    calibrator = fit_calibrator(oof["referable"], oof["raw"])
    print(f"operating point chosen out-of-fold: cut {point['cut']:.3f} "
          f"-> sensitivity {point['sensitivity']:.3f}, specificity {point['specificity']:.3f}")

    test = load_sweep_predictions(cfg, sweep)
    test["referable"] = test["diagnosis"] >= REFERABLE_FROM

    externals = {}
    for key in external.DATASETS:
        ds = external.dataset(key)
        print(f"scoring {ds.name}:")
        try:
            frame = external.score(cfg, sweep, key)
        except FileNotFoundError as exc:
            print(f"  skipped - {str(exc).splitlines()[0]}")
            continue
        frame["referable"] = frame["diagnosis"] >= REFERABLE_FROM
        externals[ds.name] = frame

    sets = {"APTOS out-of-fold (selection)": oof, "APTOS test": test, **externals}

    lines = [
        "# Calibration and the clinical operating point",
        "",
        "Generated by `python -m aptos.evaluation.calibration`.",
        "",
        f"Operating point selected on **out-of-fold** predictions at "
        f"sensitivity >= {args.min_sensitivity:.2f}, then applied unchanged to the "
        f"held-out test split and to the external sets. Selecting on test and then reporting "
        f"test performance would measure the selection, not the model.",
        "",
        f"Chosen cut: **{point['cut']:.3f}** on the raw ordinal score.",
        "",
        "## Referral decision at that operating point",
        "",
    ]

    rows = []
    for name, frame in sets.items():
        applied = apply_operating_point(frame["referable"], frame["raw"], point["cut"])
        rows.append({
            "set": name, "n": len(frame),
            "prevalence": float(frame["referable"].mean()),
            "sensitivity": applied["sensitivity"],
            "specificity": applied["specificity"],
            "PPV": applied["ppv"],
            "referral rate": applied["referral_rate"],
            "ROC AUC": applied["roc_auc"],
        })
    lines += _md_table(pd.DataFrame(rows)) + [""]

    # Computed rather than written in: this sentence used to quote IDRiD's
    # numbers as literal text, which could not cover a second external set.
    from aptos.modeling.thresholds import apply_thresholds

    _, thresholds = external.load_ensemble(sweep)
    top, boundary = [], []
    for name, frame in externals.items():
        pred = apply_thresholds(frame["raw"].to_numpy(), thresholds)
        true = frame["diagnosis"].to_numpy()
        top.append(f"{name} {int((pred == 4).sum())} grade-4 predictions where "
                   f"{int((true == 4).sum())} exist")
        moderate = true == 2
        if moderate.any():
            boundary.append(f"{name} {(pred[moderate] < 2).mean():.0%} of "
                            f"{int(moderate.sum())}")
    lines += [
        "Where the grade goes wrong decides whether the referral does. At the top "
        "of the scale the external sets show " + ("; ".join(top) or "nothing yet")
        + " - compressing grades 3 and 4 against each other costs no referrals. "
        "At the referral boundary, the share of true grade-2 (Moderate) eyes "
        "graded below 2 is " + ("; ".join(boundary) or "not measured")
        + ". Those are missed referrals.",
        "",
        "## Calibration",
        "",
        "The model emits an ordinal severity score, not a probability, so there "
        "is nothing to temperature-scale. Platt scaling on the out-of-fold "
        "predictions is the equivalent step.",
        "",
    ]

    for name, frame in sets.items():
        prob = probabilities(calibrator, frame["raw"])
        ece, table = expected_calibration_error(frame["referable"], prob)
        gap = calibration_in_the_large(frame["referable"], prob)
        verdict = ("well calibrated on average" if abs(gap) < 0.01
                   else "under-confident" if gap > 0 else "over-confident")
        lines += [f"### {name} - ECE {ece:.4f}", "",
                  f"Observed referable rate minus mean predicted probability: "
                  f"{gap:+.4f} ({verdict}).", ""]
        lines += _md_table(table) + [""]

    transfer = []
    names = list(externals)
    if len(names) >= 2:
        lines += [
            "## Recalibrating at a new site",
            "",
            "Labels from a new site are scarce. So: refit the calibrator and the "
            "operating point on one external set, then judge them on the other. "
            "If that helps in both directions, one local recalibration transfers; "
            "if not, every site needs its own.",
            "",
        ]
        for fit_name in names:
            for test_name in names:
                if fit_name == test_name:
                    continue
                result = site_transfer(externals[fit_name], externals[test_name],
                                       calibrator, point["cut"], args.min_sensitivity)
                transfer.append({"fitted on": fit_name, "applied to": test_name, **result})
        lines += _md_table(pd.DataFrame(transfer)) + [""]

    lines += ["## Decision curve", "",
              "Net benefit against treating everyone and treating no one. A model "
              "worth using has to beat both across the range of thresholds a "
              "clinician might hold.", ""]
    for name, frame in sets.items():
        prob = probabilities(calibrator, frame["raw"])
        curve = decision_curve(frame["referable"], prob)
        beats = curve["model_beats_both"].mean()
        lines += [f"### {name} - beats both policies at "
                  f"{beats:.0%} of thresholds tested", ""]
        lines += _md_table(curve.drop(columns=["model_beats_both"]), floats=4) + [""]

    out = cfg.paths.root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out.with_suffix(".json")).write_text(
        json.dumps({"operating_point": point, "applied": rows, "transfer": transfer},
                   indent=2), encoding="utf-8"
    )
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
