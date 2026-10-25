"""Retrieve the pinned author-released ADCIS pairing transcription, locally only."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

REVISION = "b8c9322863b5e2d79df964e524d74ad53d1d1b7a"
URL = f"https://raw.githubusercontent.com/ok1zjf/RFM/{REVISION}/datasets/messidor2_pairs.csv"
SHA256 = "d52696ff9e289feb9cf08a7f1ccc5e9a8f63bb5d6a305cc55f5b873fa06d22d3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("data/external/messidor2/messidor2_pairs.csv"))
    args = parser.parse_args()
    with urlopen(URL, timeout=60) as response:
        content = response.read(100_001)
    if len(content) > 100_000 or hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError("pair table size/hash differs from audited source")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(content)
    print(f"Saved verified author-released transcription to {args.out}")


if __name__ == "__main__":
    main()
