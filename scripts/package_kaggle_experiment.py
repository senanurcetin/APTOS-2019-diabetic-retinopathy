"""Build a token-free, private Kaggle GPU kernel bundle for each research run.

The bundle is local until the researcher explicitly passes --push. The official
Kaggle CLI reads KAGGLE_API_TOKEN or an authorized local token file; nothing
in this program prints, handles or serializes the credential.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path


def make_bundle(
    repo: Path, *, owner: str, mode: str, seed: int,
    execute: bool = False, output_root: Path | None = None,
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", owner):
        raise ValueError("owner must be an explicit Kaggle account slug")
    if mode not in ("reg", "ordinal") or isinstance(seed, bool) or seed < 0:
        raise ValueError("mode must be reg/ordinal and seed must be nonnegative")
    original = repo / "notebooks" / "ordinal_experiment_matrix.ipynb"
    notebook = json.loads(original.read_text(encoding="utf-8"))
    assert notebook.get("nbformat") == 4
    changes = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        if "RUN_EXPERIMENTS = False" in text:
            text = text.replace("RUN_EXPERIMENTS = False",
                                f"RUN_EXPERIMENTS = {execute}")
            changes += 1
        if 'MODE = "reg"' in text:
            text = text.replace('MODE = "reg"', f'MODE = "{mode}"')
            changes += 1
        if "SEED = 42" in text:
            text = text.replace("SEED = 42", f"SEED = {seed}")
            changes += 1
        cell["source"] = text.splitlines(keepends=True)
        cell["outputs"], cell["execution_count"] = [], None
    if changes != 3:
        raise ValueError("notebook missing run/mode/seed research controls")

    target = (output_root or repo / "outputs" / "kaggle_research") / f"{mode}-seed{seed}"
    target.mkdir(parents=True, exist_ok=True)
    code_file = "ordinal_experiment_matrix.ipynb"
    (target / code_file).write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    slug = f"aptos-research-{mode}-seed{seed}"
    metadata = {
        "id": f"{owner}/{slug}",
        "title": f"APTOS research {mode} seed {seed}",
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["mariaherrerot/aptos2019"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--owner", required=True, help="Kaggle username slug")
    parser.add_argument("--mode", choices=["reg", "ordinal"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run", action="store_true",
                        help="enable GPU training inside generated notebook")
    parser.add_argument("--push", action="store_true",
                        help="explicitly upload via the local Kaggle CLI; disabled by default")
    args = parser.parse_args(argv)
    bundle = make_bundle(args.repo_root, owner=args.owner, mode=args.mode,
                         seed=args.seed, execute=args.run)
    print(f"local private kernel bundle: {bundle}")
    if args.push:
        if not shutil.which("kaggle"):
            raise FileNotFoundError("Install the official Kaggle CLI first: pip install kaggle")
        command = ["kaggle", "kernels", "push", "-p", str(bundle),
                   "--accelerator", "NvidiaTeslaT4"]
        if not args.run:
            command.append("--no-run")
        # Never add tokens to argv, notebook metadata or process output.
        subprocess.run(command, check=True)
        print(f"Kaggle upload requested for {args.mode} seed {args.seed}; check its run status")
    else:
        print("No network action taken; use --push --run only on an authorized machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
