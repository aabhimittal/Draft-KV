#!/usr/bin/env python3
"""Download the open-source checkpoint the real-model experiments use.

    python scripts/fetch_model.py            # distilgpt2 -> ~/.cache/draftkv/distilgpt2
    python scripts/fetch_model.py --model gpt2

No torch, no transformers: the weights are read straight from safetensors by
`draftkv.gpt2`.  Tests and benchmarks that need the checkpoint skip themselves
when it is absent, so this is optional.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

FILES = ["config.json", "model.safetensors", "vocab.json", "merges.txt"]
BASE = "https://huggingface.co/{model}/resolve/main/{f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--dest", default=None)
    a = ap.parse_args()

    dest = Path(a.dest) if a.dest else Path.home() / ".cache" / "draftkv" / a.model.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        out = dest / f
        if out.exists():
            print(f"have {f}")
            continue
        url = BASE.format(model=a.model, f=f)
        print(f"get  {f} ...", end="", flush=True)
        urllib.request.urlretrieve(url, out)
        print(f" {out.stat().st_size / 1e6:.1f} MB")
    print(f"\n{dest}\nexport DRAFTKV_GPT2_DIR={dest}")


if __name__ == "__main__":
    main()
