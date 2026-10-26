"""Package a pinned private Kaggle follow-up using attached data and base outputs."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

BOOTSTRAP = '''from pathlib import Path
import hashlib, json, subprocess, sys

SOURCE_COMMIT = "__COMMIT__"
CODE = Path("/kaggle/temp/aptos-followup-code")
RUNTIME = Path("/kaggle/temp/aptos-followup-runtime")
OUTPUT = Path("/kaggle/working/examination-followup-v2")
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "scikit-learn==1.8.0", "pandas==2.2.3", "timm", "pyyaml"], check=True)
CODE.mkdir(parents=True, exist_ok=False)
for command in (["git", "init", str(CODE)], ["git", "-C", str(CODE), "remote", "add", "origin", "https://github.com/senanurcetin/APTOS-2019-diabetic-retinopathy.git"], ["git", "-C", str(CODE), "fetch", "--depth", "1", "origin", SOURCE_COMMIT], ["git", "-C", str(CODE), "checkout", "--detach", "FETCH_HEAD"]):
    subprocess.run(command, check=True)
sys.path.insert(0, str(CODE / "src"))
import torch
assert torch.cuda.is_available(), "No free CUDA runtime available; stopping"
print("SOURCE_COMMIT", SOURCE_COMMIT, "GPU", torch.cuda.get_device_name(0), flush=True)
from aptos.evaluation.examination_split import make_examination_split
from aptos.evaluation.external_label_integrity import validate_external_labels
from aptos.training.examination_followup import align_source_filenames, digest, run
import pandas as pd

plan_path = CODE / "reports/research/messidor2_examination_v2_plan.json"
plan = json.loads(plan_path.read_text())
sources = json.loads((CODE / "reports/research/messidor2_sources.json").read_text())
grading = [p for p in Path("/kaggle/input").rglob("messidor_data.csv") if digest(p) == plan["inputs_sha256"]["labels"]]
assert len(grading) == 1, "Missing or ambiguous pinned Google labels"
external = RUNTIME / "data/external/messidor2"
external.mkdir(parents=True)
images = [p for p in Path("/kaggle/input").rglob("preprocess") if p.is_dir() and len([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg")]) == 1744]
assert len(images) == 1, "Missing or ambiguous Messidor image directory"
tree_hash = hashlib.sha256()
for path in sorted(images[0].iterdir()):
    if path.is_file():
        tree_hash.update(path.name.encode()); tree_hash.update(b"\\0"); tree_hash.update(bytes.fromhex(digest(path)))
assert tree_hash.hexdigest() == sources["image_tree_sha256"], "Source image bytes differ from audited mirror"
source_grades = pd.read_csv(grading[0]); source_grades = source_grades[source_grades.adjudicated_gradable.eq(1)]
align_source_filenames(source_grades, images[0]).to_csv(external / "messidor_data.csv", index=False)
destination = external / "messidor-2/messidor-2"
destination.mkdir(parents=True)
(destination / "preprocess").symlink_to(images[0], target_is_directory=True)
pair_path = external / "messidor2_pairs.csv"
subprocess.run([sys.executable, str(CODE / "scripts/fetch_messidor2_pairs.py"), "--out", str(pair_path)], check=True)
grades = pd.read_csv(grading[0]); grades = grades[grades.adjudicated_gradable.eq(1)]
grades = grades.rename(columns={"image_id":"id_code", "adjudicated_dr_grade":"diagnosis"})
grades = validate_external_labels(grades, source="Google", filenames=True)
split = make_examination_split(grades, pd.read_csv(pair_path, sep=";"), seed=42)
split_path = RUNTIME / "frozen_split.csv"
split.to_csv(split_path, index=False, lineterminator="\\n")
assert digest(split_path) == plan["split_csv_sha256"], "Software-dependent split drift; do not train"
sweeps = [p.parent for p in Path("/kaggle/input").rglob("summary.json") if p.parent.name == "baseline-c8232af68337"]
assert len(sweeps) == 1, "Attach the completed seed42 kernel output"
run(sweeps[0], split_path, plan_path, pair_path, RUNTIME, OUTPUT, CODE)
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    changed = subprocess.check_output(["git", "-C", str(args.repo), "status", "--porcelain"], text=True)
    if changed.strip():
        parser.error("commit the protocol and runner before packaging a pinned GPU experiment")
    commit = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "followup.py").write_text(BOOTSTRAP.replace("__COMMIT__", commit), encoding="utf-8")
    metadata = {
        "id": "senanuretin/aptos-messidor-examination-followup-v2",
        "title": "APTOS Messidor Examination Followup V2",
        "code_file": "followup.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["google-brain/messidor2-dr-grades", "mariaherrerot/messidor2preprocess"],
        "kernel_sources": ["senanuretin/aptos-research-reg-seed42"],
        "competition_sources": [], "model_sources": [],
    }
    (args.output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Private bundle ready, source {commit}: {args.output}")


if __name__ == "__main__":
    main()
