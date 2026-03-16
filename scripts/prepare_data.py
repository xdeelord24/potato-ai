"""
Prepare training data for Potato AI.
- Trains a SentencePiece tokenizer on your text data
- Optionally downloads public datasets (requires datasets package)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def train_tokenizer(data_dir: Path, output_path: Path, vocab_size: int = 32000) -> None:
    """Train SentencePiece tokenizer on all .txt files in data_dir."""
    import sentencepiece as spm

    data_dir = Path(data_dir)
    output_path = Path(output_path)
    txt_files = list(data_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files in {data_dir}")

    input_path = data_dir / "_corpus.txt"
    with open(input_path, "w", encoding="utf-8") as f:
        for p in txt_files:
            f.write(p.read_text(encoding="utf-8", errors="ignore"))
            f.write("\n\n")

    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(output_path.with_suffix("")),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="<|pad|>",
        bos_piece="<|bos|>",
        eos_piece="<|eos|>",
        unk_piece="<|unk|>",
    )
    input_path.unlink(missing_ok=True)
    (Path(str(output_path.with_suffix("")) + ".vocab")).unlink(missing_ok=True)
    print(f"Tokenizer saved to {output_path.with_suffix('')}.model")


def download_wikitext(output_dir: Path, split: str = "train") -> None:
    """Download WikiText-2 (small) for additional pre-training data."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    out_path = output_dir / "wikitext.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            text = ex.get("text", "").strip()
            if text and not text.startswith("="):
                f.write(text + "\n")
    print(f"Saved {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--tokenizer_out", type=Path, default=Path("data/tokenizer.model"))
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--download_wikitext", action="store_true", help="Download WikiText-2")
    args = parser.parse_args()

    if args.download_wikitext:
        download_wikitext(args.data_dir)

    txt_files = list(args.data_dir.glob("*.txt"))
    if txt_files:
        # SentencePiece caps vocab at ~corpus unique tokens; use 4096 max for small data
        vocab_size = min(args.vocab_size, 4096)
        train_tokenizer(args.data_dir, args.tokenizer_out, vocab_size)
    else:
        print("No .txt files found. Add files to data/ or use --download_wikitext")


if __name__ == "__main__":
    main()
