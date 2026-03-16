"""Validate the SSM backbone with a tiny model (~10M params)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.ssm import PotatoConfig, PotatoLM


def main() -> None:
    config = PotatoConfig.tiny()
    model = PotatoLM(config)

    print(f"Model parameters: {model.num_parameters:,}")

    # Test forward (batch_size=1 for step mode - cache assumes single sequence)
    batch_size = 1
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, cache = model(input_ids, None)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    print(f"Forward pass: logits shape {logits.shape}")

    # Test step mode (single token)
    # Cache from full forward is for continuing; must pass exactly 1 token
    single_token = torch.randint(0, config.vocab_size, (batch_size, 1))
    assert all(c is not None for c in cache), "Cache should be populated"
    logits_step, _ = model(single_token, cache)
    assert logits_step.shape == (batch_size, 1, config.vocab_size)
    print("Step mode: OK")

    # Test generate
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=5, temperature=0.8)
    assert generated.shape[1] >= 11, f"Expected >= 11 tokens, got {generated.shape[1]}"
    print(f"Generate: {prompt.shape} -> {generated.shape}")

    print("\nAll validation passed!")


if __name__ == "__main__":
    main()
