"""NEXUS Identity 2.0 branding is the exact source SVG, not a substitute.

These are pure structural contracts; no Kaggle account, images, GPU or browser.
"""
import base64
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("research_evidence.ipynb", "ordinal_experiment_matrix.ipynb")
SOURCE = ROOT / "serving" / "static" / "nexus-logo.svg"
X_MARK = ROOT / "serving" / "static" / "nexus-x.svg"


def _git_blob_sha(payload: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\x00" + payload).hexdigest()


@pytest.mark.pure
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_official_wordmark_attached_to_portable_notebook(name):
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    first = notebook["cells"][0]
    assert first["cell_type"] == "markdown"
    assert "attachment:nexus-official-logo.svg" in "".join(first["source"])
    encoded = first["attachments"]["nexus-official-logo.svg"]["image/svg+xml"]
    assert base64.b64decode(encoded, validate=True) == SOURCE.read_bytes()
    assert notebook["metadata"]["nexus_identity"]["scope"] == "research_only"
    assert all(cell["execution_count"] is None and not cell["outputs"]
               for cell in notebook["cells"] if cell["cell_type"] == "code")


@pytest.mark.pure
def test_official_drive_wordmark_and_icon_object_hashes():
    assert _git_blob_sha(SOURCE.read_bytes()) == "7913f5155f6620ed771a602db409cf428a347c94"
    assert _git_blob_sha(X_MARK.read_bytes()) == "5872cf6339bc31f4c11f2699779a9f3e582f9d64"


@pytest.mark.pure
def test_evidence_plot_colors_are_source_identity_palette_only():
    nb = json.loads((ROOT / "notebooks" / "research_evidence.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for hex_color in ("#0B0F19", "#00E0FF", "#4EA8FF", "#8F9BFF", "#6E55C6"):
        assert hex_color in code
    assert "from aptos.reporting.research_evidence import build_evidence" in code
    assert "sensitivity_wilson95" in code
    assert "ax.set_ylim(0, 1)" in code


@pytest.mark.pure
def test_research_demo_loads_the_official_assets():
    page = (ROOT / "serving" / "static" / "index.html").read_text()
    assert 'src="/static/nexus-logo.svg"' in page
    assert 'href="/static/nexus-x.svg"' in page
    assert 'href="/static/nexus-theme.css"' in page
