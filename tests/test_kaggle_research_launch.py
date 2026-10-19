"""Local-only Kaggle staging and bundle metadata tests; no Kaggle account needed."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "kaggle"
    checkout = tmp_path / "checkout"
    (checkout / "configs").mkdir(parents=True)
    (checkout / "configs" / "cv.yaml").write_text("cv: {}")
    sample = {"train": [("a", 0), ("b", 1)],
              "valid": [("c", 2)], "test": [("d", 3), ("e", 4)]}
    mapping = {"train": ("train_1.csv", "train_images"),
               "valid": ("valid.csv", "val_images"),
               "test": ("test.csv", "test_images")}
    for part, rows in sample.items():
        csvname, folder = mapping[part]
        src = root / folder / folder
        src.mkdir(parents=True)
        for ident, _ in rows:
            (src / f"{ident}.png").write_bytes(b"not-a-real-fundus-image")
        with (root / csvname).open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["id_code", "diagnosis"])
            writer.writerows(rows)
    return root, checkout


@pytest.mark.pure
def test_kaggle_stage_is_repeatable_and_only_links_source_images(dataset):
    prep = load_script("kaggle_dataset_preflight")
    mounted, repo = dataset
    dry = prep.stage_dataset(mounted, repo, expected_rows=5, dry_run=True)
    assert dry["per_split_counts"] == {"train": 2, "valid": 1, "test": 2}
    assert not (repo / "data").exists()
    receipt = prep.stage_dataset(mounted, repo, expected_rows=5)
    assert receipt == dry
    assert (repo / "data" / "images" / "val_images" / "val_images").is_symlink()
    assert (repo / "data" / "raw" / "train_1.csv").exists()
    prep.stage_dataset(mounted, repo, expected_rows=5)
    assert (repo / "outputs" / "kaggle_dataset_preflight.json").exists()


@pytest.mark.pure
def test_refuses_missing_image_before_creating_artifacts(dataset):
    prep = load_script("kaggle_dataset_preflight")
    mounted, repo = dataset
    (mounted / "train_images" / "train_images" / "b.png").unlink()
    with pytest.raises(ValueError, match="expected one PNG"):
        prep.stage_dataset(mounted, repo, expected_rows=5)
    assert not (repo / "data").exists()


@pytest.mark.pure
def test_refuses_fractional_grade(dataset):
    prep = load_script("kaggle_dataset_preflight")
    mounted, repo = dataset
    target = mounted / "valid.csv"
    target.write_text("id_code,diagnosis\nc,2.7\n")
    with pytest.raises(ValueError, match="invalid eye ID or grade"):
        prep.stage_dataset(mounted, repo, expected_rows=5)


@pytest.mark.pure
def test_kaggle_bundle_is_private_and_never_embeds_token(tmp_path):
    bundle = load_script("package_kaggle_experiment")
    saved = bundle.make_bundle(REPO, owner="senanuretin", mode="ordinal", seed=43,
                               execute=True, output_root=tmp_path)
    metadata = json.loads((saved / "kernel-metadata.json").read_text())
    nb = json.loads((saved / metadata["code_file"]).read_text())
    body = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert metadata["enable_gpu"] and metadata["is_private"]
    assert metadata["dataset_sources"] == ["mariaherrerot/aptos2019"]
    assert "RUN_EXPERIMENTS = True" in body
    assert 'MODE = "ordinal"' in body and "SEED = 43" in body
    assert "KGAT_" not in body and "KAGGLE_API_TOKEN" not in body
    assert all(c["outputs"] == [] and c["execution_count"] is None
               for c in nb["cells"] if c["cell_type"] == "code")


@pytest.mark.pure
def test_packaging_dry_run_keeps_training_off(tmp_path):
    bundle = load_script("package_kaggle_experiment")
    saved = bundle.make_bundle(REPO, owner="senanuretin", mode="reg", seed=42,
                               output_root=tmp_path)
    text = (saved / "ordinal_experiment_matrix.ipynb").read_text()
    assert "RUN_EXPERIMENTS = False" in text


@pytest.mark.pure
def test_rejects_invalid_owners_or_modes(tmp_path):
    bundle = load_script("package_kaggle_experiment")
    with pytest.raises(ValueError, match="owner"):
        bundle.make_bundle(REPO, owner="../../evil", mode="reg", seed=42,
                           output_root=tmp_path)
    with pytest.raises(ValueError, match="mode"):
        bundle.make_bundle(REPO, owner="owner", mode="cls", seed=42,
                           output_root=tmp_path)
