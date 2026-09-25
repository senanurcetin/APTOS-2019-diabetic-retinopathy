"""Single-split training - moved into the package.

The implementation is now `aptos.training.single`. This file forwards to it
with the same command line, because scripts/run_seeds.sh and older notes call
it by path.

    python scripts/train.py --data-dir data/processed --seed 42 --exclude-leaked
    python -m aptos.training.single ...   # equivalent
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aptos.training.single import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
