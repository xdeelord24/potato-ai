"""
Run Potato AI inference (interactive or one-shot).
"""

import argparse
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.engine import InferenceEngine
from src.ssm import PotatoConfig, PotatoLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint path")
    parser.add_argument("--config", default="tiny", choices=["tiny", "small", "base"])
    parser.add_argument("--vocab_size", type=int, default=None, help="Override vocab size (required if trained with custom tokenizer)")
    parser.add_argument("--prompt", type=str, help="One-shot prompt")
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    configs = {"tiny": PotatoConfig.tiny(), "small": PotatoConfig.small(), "base": PotatoConfig.base()}
    config = configs[args.config]
    if args.vocab_size is not None:
        config.vocab_size = ((args.vocab_size + 15) // 16) * 16

    if args.checkpoint and args.checkpoint.exists():
        engine = InferenceEngine(config=config, checkpoint_path=args.checkpoint)
    else:
        model = PotatoLM(config)
        engine = InferenceEngine(model=model)

    # Use tokenizer if available
    tokenizer_path = Path("data") / "tokenizer.model"
    if tokenizer_path.exists():
        from src.tokenizer import PotatoTokenizer
        tok = PotatoTokenizer()
        tok.load(tokenizer_path)
        encode_fn = lambda t: tok.encode(t, add_bos=True)
        decode_fn = lambda ids: tok.decode(ids)
    else:
        encode_fn = lambda t: [min(ord(c), 31999) for c in t[:1024]]
        decode_fn = lambda ids: "".join(chr(i) if 32 <= i < 65536 else "?" for i in ids)

    if args.prompt:
        ids = encode_fn(args.prompt)
        result = engine.complete(
            prompt_ids=ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(decode_fn(result))
    else:
        print("Potato AI - type a prompt and press Enter (empty to exit)")
        while True:
            prompt = input("> ").strip()
            if not prompt:
                break
            ids = encode_fn(prompt)
            result = engine.complete(
                prompt_ids=ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(decode_fn(result))
            print()


if __name__ == "__main__":
    main()
