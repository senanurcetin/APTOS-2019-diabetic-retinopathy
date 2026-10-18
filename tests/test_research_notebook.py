"""Notebook compatibility checks without Jupyter or a GPU."""
import ast
import json
from pathlib import Path

import pytest


@pytest.mark.pure
def test_research_notebook_has_executable_clean_cells():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "research_evidence.ipynb"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert data["metadata"]["kernelspec"]["language"] == "python"
    cells = data["cells"]
    assert len(cells) >= 7
    assert "not a medical device" in "".join(cells[0]["source"])
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert len(code_cells) >= 3
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        ast.parse("".join(cell["source"]))
