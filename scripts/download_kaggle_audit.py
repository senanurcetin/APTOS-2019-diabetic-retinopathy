"""Download paginated Kaggle audit outputs; exclude images and pickle weights."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True, help="owner/kernel-slug")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", args.kernel):
        parser.error("kernel must be owner/slug")
    # The optional official CLI reads credentials from its normal local store.
    import requests
    from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest, KaggleApi

    api = KaggleApi()
    api.authenticate()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    owner, slug = args.kernel.split("/")
    inventory, seen, token, pages = [], set(), None, 0
    pattern = re.compile(r"/models/cv/[^/]+/(config|provenance|summary|fold[1-5])\.(json|npz)$")
    with api.build_kaggle_client() as client:
        for _ in range(100):
            pages += 1
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = owner, slug
            if token:
                request.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in response.files or []:
                inventory.append({"name": item.file_name})
                if not pattern.search(item.file_name):
                    continue
                destination = (root / item.file_name).resolve()
                if not destination.is_relative_to(root):
                    raise ValueError("output path escapes destination")
                download = requests.get(item.url, timeout=60)
                download.raise_for_status()
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(download.content)
                print(f"Downloaded {item.file_name}")
            if response.log:
                (root / "execution.log").write_text(response.log, encoding="utf-8")
            token = response.next_page_token
            if not token:
                break
            if token in seen:
                raise RuntimeError("Kaggle repeated an output pagination token")
            seen.add(token)
        else:
            raise RuntimeError("output inventory exceeded bounded pagination")
    (root / "inventory.json").write_text(
        json.dumps({"kernel": args.kernel, "pages": pages, "files": inventory}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Inventoried {len(inventory)} files across {pages} pages; no images/weights downloaded")


if __name__ == "__main__":
    main()
