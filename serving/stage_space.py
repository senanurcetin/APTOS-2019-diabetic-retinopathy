"""Assemble the directory that becomes the Hugging Face Space.

The Space is a copy of the serving subset, not the repository: the package, the
configs, the service, one sweep's fold checkpoints, a Dockerfile and a README
that doubles as the model card. Staging it with a script rather than by hand
means the deployed artefact can be rebuilt exactly, and checked, before
anything is uploaded.

    python serving/stage_space.py --sweep models/cv/baseline-<id> --out ../aptos-space
"""
from __future__ import annotations

import argparse
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Only what the service imports. data/, reports/, mlflow.db and the training
# scripts stay behind - nothing in them is needed to serve, and nothing in them
# should be published by accident.
COPY = ["src", "configs", "pyproject.toml"]
SERVING = ["app.py"]


def stage(sweep: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for name in COPY:
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, out / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        else:
            shutil.copy2(src, out / name)

    (out / "serving").mkdir()
    for name in SERVING:
        shutil.copy2(ROOT / "serving" / name, out / "serving" / name)

    checkpoints = sorted(sweep.glob("fold*.pt"), key=lambda p: int(p.stem[4:]))
    if not checkpoints:
        raise FileNotFoundError(f"no fold checkpoints under {sweep}")
    target = out / "models" / "cv" / sweep.name
    target.mkdir(parents=True)
    for ckpt in checkpoints:
        shutil.copy2(ckpt, target / ckpt.name)

    shutil.copy2(ROOT / "serving" / "Dockerfile", out / "Dockerfile")
    shutil.copy2(ROOT / "serving" / "SPACE_README.md", out / "README.md")

    # Checkpoints are binary and large; Spaces store them through Git LFS.
    (out / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n",
                                        encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--out", default=str(ROOT.parent / "aptos-space"))
    args = parser.parse_args()

    out = stage(pathlib.Path(args.sweep).resolve(), pathlib.Path(args.out).resolve())
    files = sorted(p for p in out.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"staged {len(files)} files, {total / 1e6:.1f} MB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
