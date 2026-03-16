"""
Supervised fine-tuning (instruction tuning) for Potato AI.
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
    """Single SFT step."""
    model.train()
    input_ids = batch["input_ids"].to(device)
    logits, _ = model(input_ids, None)
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
    parser.add_argument("--checkpoint", type=Path, help="Pre-trained checkpoint")
    parser.add_argument("--config", default="tiny", choices=["tiny", "small", "base"])
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    configs = {"tiny": PotatoConfig.tiny(), "small": PotatoConfig.small(), "base": PotatoConfig.base()}
    config = configs[args.config]
    model = PotatoLM(config).to(args.device)

    if args.checkpoint and args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"Loaded {args.checkpoint}")

    # Sample instruction data
    examples = [
        {"instruction": "What is 2+2?", "output": "2+2 equals 4."},
        {"instruction": "Say hello.", "output": "Hello! How can I help you?"},
    ]

    class DummyTokenizer:
        def encode(self, text, add_bos=False, add_eos=False):
            return [min(ord(c), 31999) for c in text[:512]]

    from data.loaders import InstructionDataset
    dataset = InstructionDataset(examples, DummyTokenizer(), max_length=128)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for step, batch in enumerate(loader):
        if step >= args.max_steps:
            break
        loss = train_step(model, batch, optimizer, args.device)
        if step % 10 == 0:
            print(f"Step {step} loss={loss:.4f}")

    torch.save(model.state_dict(), args.output_dir / f"sft_{args.config}.pt")
    print("Done.")


if __name__ == "__main__":
    main()
