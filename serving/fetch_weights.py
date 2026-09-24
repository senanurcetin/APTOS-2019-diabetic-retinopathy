"""Download the ONNX export from its Hugging Face model repository.

Used at image build time, and runnable locally, so the build step is something
that has been executed rather than something that is hoped to work. Standard
library only: it runs before anything else is installed.

    python serving/fetch_weights.py --repo senanurcetin/aptos-retinopathy-grader \
        --revision <commit> --sweep baseline-<id> --out models/onnx
"""
from __future__ import annotations

import argparse
import json
import pathlib
import urllib.request

FILES = ["export.json"] + [f"fold{i}.onnx" for i in range(1, 6)]


def fetch(repo: str, revision: str, sweep: str, out: pathlib.Path) -> pathlib.Path:
    target = out / sweep
    target.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{repo}/resolve/{revision}"
    for name in FILES:
        path = target / name
        urllib.request.urlretrieve(f"{base}/{name}", path)
        print(f"  {name:12s} {path.stat().st_size:>12,d} bytes")

    meta = json.loads((target / "export.json").read_text(encoding="utf-8"))
    if meta.get("sweep") != sweep:
        raise RuntimeError(f"export.json is for {meta.get('sweep')!r}, expected {sweep!r}")
    parity = meta.get("parity", {})
    if parity.get("grade_mismatches", 1) != 0:
        raise RuntimeError("this export failed its parity check and must not be served")
    print(f"  parity recorded at export: max diff {parity['max_abs_diff_raw']:.1e}, "
          f"{parity['grade_mismatches']} grade mismatches")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--out", default="models/onnx")
    args = parser.parse_args()
    fetch(args.repo, args.revision, args.sweep, pathlib.Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
