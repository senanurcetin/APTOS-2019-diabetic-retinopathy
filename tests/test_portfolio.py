"""Research explorer must preserve evidence and reject stale/broken artifacts."""
import json
import shutil
from pathlib import Path

import pytest

from aptos.reporting.portfolio import build_payload, render

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.pure


def test_two_operating_points_remain_distinct():
    payload = build_payload(ROOT / "reports")
    external = {row["dataset"]: row for row in payload["external"]}
    operating = {row["cohort"]: row for row in payload["operating_points"]}
    assert external["Messidor-2"]["missed_referable"] == 307
    assert operating["Messidor-2"]["fn"] == 338
    assert external["IDRiD"]["sensitivity"] > operating["IDRiD"]["sensitivity"]
    assert len(payload["inputs_sha256"]) == 4


def test_generated_page_is_current():
    assert (ROOT / "serving/static/research.html").read_text(encoding="utf-8") == render(ROOT)


def test_corrupt_reports_cannot_be_published(tmp_path):
    shutil.copytree(ROOT / "reports", tmp_path / "reports")
    path = tmp_path / "reports/external_validation_messidor2.json"
    report = json.loads(path.read_text())
    report["ref_fn"] += 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        build_payload(tmp_path / "reports")
