"""Paired uncertainty for ordinal QWK and fixed-five-grade macro-F1.

No model fitting, clinical operating-point tuning or patient identifiers in
output. Patient clusters are sampled together only with real patient linkage.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

from aptos.evaluation.paired_uncertainty import _grades


def _ordinal_scores(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Grade metrics with the same five-grade definition as model reports."""
    return {
        "qwk": float(cohen_kappa_score(truth, prediction, labels=list(range(5)),
                                       weights="quadratic")),
        "macro_f1": float(f1_score(truth, prediction, labels=list(range(5)),
                                   average="macro", zero_division=0)),
    }


def paired_grade_bootstrap(
    diagnosis,
    baseline_grade,
    candidate_grade,
    *,
    unit: str,
    patient_ids=None,
    n_resamples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    """Fixed-prediction paired percentile intervals, candidate minus baseline.

    QWK is undefined on some near-degenerate bootstrap resamples. Such draws
    are omitted *only* from the QWK interval, and their count is recorded.
    If fewer than 80% yield defined QWK, suppress that interval entirely.
    Macro-F1 uses all draws, with a fixed 0..4 label space.
    """
    truth = _grades(diagnosis, "diagnosis")
    baseline = _grades(baseline_grade, "baseline_grade")
    candidate = _grades(candidate_grade, "candidate_grade")
    n = len(truth)
    if len(baseline) != n or len(candidate) != n:
        raise ValueError("paired ordinal grades need matching lengths")
    if np.unique(truth).size < 2:
        raise ValueError("QWK comparison needs at least two observed grades")
    if unit not in ("eye", "patient"):
        raise ValueError("unit must be eye or patient")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 100:
        raise ValueError("at least 100 bootstrap draws required")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(confidence, (int, float)) or not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    if unit == "eye":
        if patient_ids is not None:
            raise ValueError("patient IDs supplied: select patient sampling")
        clusters = [np.array([i]) for i in range(n)]
    else:
        if patient_ids is None or len(patient_ids) != n:
            raise ValueError("patient sampling requires one patient ID per eye")
        ids = list(patient_ids)
        if any(not isinstance(v, str) or not v.strip() for v in ids):
            raise ValueError("patient IDs must be nonempty strings")
        _, inverse = np.unique(ids, return_inverse=True)
        clusters = [np.flatnonzero(inverse == k) for k in range(int(inverse.max()) + 1)]
    if len(clusters) < 2:
        raise ValueError("at least two independent units required")

    first = _ordinal_scores(truth, baseline)
    second = _ordinal_scores(truth, candidate)
    if not np.isfinite(first["qwk"]) or not np.isfinite(second["qwk"]):
        raise ValueError("QWK is undefined for this pair of complete predictions")

    draws = {"qwk": [], "macro_f1": []}
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        choices = rng.integers(0, len(clusters), size=len(clusters))
        rows = np.concatenate([clusters[int(i)] for i in choices])
        a = _ordinal_scores(truth[rows], baseline[rows])
        b = _ordinal_scores(truth[rows], candidate[rows])
        for metric in draws:
            delta = b[metric] - a[metric]
            if np.isfinite(delta):
                draws[metric].append(float(delta))

    alpha = (1 - confidence) / 2
    metrics = {}
    for name, differences in draws.items():
        sufficient = len(differences) >= max(50, int(n_resamples * 0.8))
        lower, upper = (np.quantile(differences, [alpha, 1 - alpha])
                        if sufficient else (None, None))
        metrics[name] = {
            "baseline": first[name],
            "candidate": second[name],
            "difference": second[name] - first[name],
            "ci_lower": float(lower) if lower is not None else None,
            "ci_upper": float(upper) if upper is not None else None,
            "valid_draws": len(differences),
        }
    return {
        "schema_version": 1,
        "analysis_unit": unit,
        "n_eyes": n,
        "n_units": len(clusters),
        "confidence": float(confidence),
        "seed": seed,
        "draws_requested": n_resamples,
        "method": "paired percentile bootstrap on frozen ordinal predictions",
        "metrics": metrics,
        "limitations": [
            "Intervals are descriptive for the sampled cohort, not clinical safety bounds.",
            "Repeated model selection on these same observations invalidates inference.",
            "Patient clustering is only valid when genuine patient IDs are supplied.",
            "Undefined bootstrap QWK draws are excluded and counted; CI is suppressed if too many.",
        ],
    }
