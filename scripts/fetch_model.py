#!/usr/bin/env python3
"""Download the open-source checkpoint the real-model experiments use.

    python scripts/fetch_model.py                                   # distilgpt2
    python scripts/fetch_model.py --model gpt2-medium               # depth scaling
    python scripts/fetch_model.py --model HuggingFaceTB/SmolLM2-135M  # RoPE+GQA+SwiGLU
    python scripts/fetch_model.py --model Qwen/Qwen2.5-0.5B

No torch, no transformers: the weights are read straight from safetensors by
`draftkv.gpt2`.  Tests and benchmarks that need the checkpoint skip themselves
when it is absent, so this is optional.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

REQUIRED = ["config.json", "model.safetensors"]
# GPT-2 family ships vocab.json + merges.txt; Llama/Qwen ship tokenizer.json.
# Try both and keep whatever the repo actually has.
OPTIONAL = ["tokenizer.json", "vocab.json", "merges.txt"]
BASE = "https://huggingface.co/{model}/resolve/main/{f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--dest", default=None)
    a = ap.parse_args()

    name = a.dest or a.model.split("/")[-1].lower()
    dest = Path(a.dest) if a.dest else Path.home() / ".cache" / "draftkv" / name
    dest.mkdir(parents=True, exist_ok=True)
    for f in REQUIRED + OPTIONAL:
        out = dest / f
        if out.exists():
            print(f"have {f}")
            continue
        url = BASE.format(model=a.model, f=f)
        print(f"get  {f} ...", end="", flush=True)
        try:
            urllib.request.urlretrieve(url, out)
        except Exception as e:                                   # noqa: BLE001
            out.unlink(missing_ok=True)
            if f in REQUIRED:
                raise
            print(" (not in this repo)")
            continue
        print(f" {out.stat().st_size / 1e6:.1f} MB")

    var = "DRAFTKV_LLAMA_DIR" if (dest / "tokenizer.json").exists() else "DRAFTKV_GPT2_DIR"
    print(f"\n{dest}\nexport {var}={dest}")


if __name__ == "__main__":
    main()
