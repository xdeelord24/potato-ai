"""
Quantize a Potato AI checkpoint for lower memory inference.
Saves in FP16 for ~50% size reduction; full 4-bit requires bitsandbytes.
"""

import argparse
from pathlib import Path

import torch

from src.ssm import PotatoConfig, PotatoLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Input checkpoint path")
    parser.add_argument("--output", type=Path, help="Output path (default: input_quantized.pt)")
    parser.add_argument("--config", default="tiny", choices=["tiny", "small", "base"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = parser.parse_args()

    configs = {"tiny": PotatoConfig.tiny(), "small": PotatoConfig.small(), "base": PotatoConfig.base()}
    config = configs[args.config]

    state = torch.load(args.input, map_location="cpu", weights_only=True)
    if args.dtype == "float16":
        state = {k: v.half() for k, v in state.items()}

    out = args.output or args.input.parent / f"{args.input.stem}_quantized.pt"
    torch.save(state, out)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Saved {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
