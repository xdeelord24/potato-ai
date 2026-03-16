"""
Causal LM pre-training for Potato AI.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.ssm import PotatoConfig, PotatoLM


def train_step(
    model: PotatoLM,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """Single training step."""
    model.train()
    input_ids = batch["input_ids"].to(device)
    logits, _ = model(input_ids, None)
    # Causal LM: predict next token
    shift_logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
    shift_labels = input_ids[:, 1:].contiguous().view(-1)
    loss = torch.nn.functional.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=0,
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tiny", choices=["tiny", "small", "base"])
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    configs = {"tiny": PotatoConfig.tiny(), "small": PotatoConfig.small(), "base": PotatoConfig.base()}
    config = configs[args.config]
    model = PotatoLM(config).to(args.device)

    # Create minimal dummy dataset if no data
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    text_files = list(data_dir.glob("*.txt"))
    if not text_files:
        sample = data_dir / "sample.txt"
        sample.write_text("The quick brown fox jumps over the lazy dog. " * 1000)
        text_files = [sample]

    # Dummy tokenizer for demo (use real tokenizer in production)
    class DummyTokenizer:
        def encode(self, text, add_bos=False, add_eos=False):
            return [min(ord(c), 31999) for c in text[:2048]]

    from data.loaders import TextDataset
    dataset = TextDataset(text_files, DummyTokenizer(), max_length=128)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            loss = train_step(model, batch, optimizer, args.device)
            step += 1
            if step % 10 == 0:
                print(f"Step {step} loss={loss:.4f}")
            if step >= args.max_steps:
                break
            if step % args.save_every == 0:
                path = args.output_dir / f"pretrain_{args.config}_step{step}.pt"
                torch.save(model.state_dict(), path)
                print(f"Saved {path}")
        if step >= args.max_steps:
            break

    torch.save(model.state_dict(), args.output_dir / f"pretrain_{args.config}_final.pt")
    print("Done.")


if __name__ == "__main__":
    main()
