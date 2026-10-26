"""Build the static NEXUS research explorer from validated committed reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aptos.reporting.operating_point_audit import reconstruct_confusion
from aptos.reporting.research_evidence import build_evidence, file_sha256


def build_payload(reports: Path) -> dict:
    payload = build_evidence(reports)
    calibration = json.loads((reports / "calibration.json").read_text(encoding="utf-8"))
    payload["operating_cut"] = calibration["operating_point"]["cut"]
    payload["operating_points"] = [reconstruct_confusion(row) for row in calibration["applied"]]
    payload["inputs_sha256"]["calibration.json"] = file_sha256(reports / "calibration.json")
    payload["grade_reports"] = [
        json.loads((reports / name).read_text(encoding="utf-8"))
        for name in ("external_validation.json", "external_validation_messidor2.json")
    ]
    result_path = reports / "research/messidor2_v2_results.json"
    audit_path = reports / "research/messidor2_v2_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit.get("status") != "PASS_PER_EYE_RECOMPUTATION"
            or audit.get("inputs_sha256", {}).get("results.json") != file_sha256(result_path)):
        raise ValueError("follow-up requires a matching completed per-eye audit")
    payload["followup"] = json.loads(result_path.read_text(encoding="utf-8"))
    for path in (result_path, audit_path):
        payload["inputs_sha256"][path.relative_to(reports).as_posix()] = file_sha256(path)
    return payload


def render(root: Path) -> str:
    template = (root / "serving/static/research.template.html").read_text(encoding="utf-8")
    if template.count("__EVIDENCE_JSON__") != 1:
        raise ValueError("research template must have exactly one evidence marker")
    encoded = json.dumps(build_payload(root / "reports"), ensure_ascii=True).replace("<", "\\u003c")
    return template.replace("__EVIDENCE_JSON__", encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = render(args.root)
    target = args.root / "serving/static/research.html"
    if args.check:
        if not target.exists() or target.read_text(encoding="utf-8") != content:
            raise SystemExit("Research explorer is stale; run python -m aptos.reporting.portfolio")
        print("Research explorer matches validated source reports")
    else:
        target.write_text(content, encoding="utf-8")
        print(f"Built {target}")


if __name__ == "__main__":
    main()
