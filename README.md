# Potato AI

Efficient language model with agent capabilities, designed for **8GB RAM, CPU-only** systems. Uses a **Mamba-2 / SSM** backbone instead of transformers for O(n) memory and constant-time per-token inference.

## Features

- **SSM backbone**: Mamba-2 (Structured State Space Duality) - linear memory, fast inference
- **Agent layer**: ReAct loop with tool use, thought/action/observation
- **8GB target**: ~1B params at 4-bit quant fits in ~600MB
- **CPU-optimized**: PyTorch inference, optional `torch.compile`

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Validate (tiny ~11M params)
python scripts/validate.py

# Run inference (no checkpoint = random init)
python scripts/run_inference.py --prompt "Hello, world"
```

## Project Structure

```
NEW-AI/
├── src/ssm/          # Mamba-2 backbone
├── inference/        # Engine, quantize, server
├── agent/            # ReAct loop, tools, parser
├── training/         # Pretrain, SFT, agent finetune
├── data/             # Loaders
└── scripts/          # Run scripts
```

## Training Pipeline

1. **Pre-train**: `python -m training.pretrain --config tiny --max_steps 1000`
2. **SFT**: `python -m training.sft --checkpoint checkpoints/pretrain_tiny_final.pt`
3. **Agent**: `python -m training.agent_finetune --checkpoint checkpoints/sft_tiny.pt`

## Configs

- `tiny`: ~11M params (validation)
- `small`: ~350M params
- `base`: ~1B params (8GB target)

## References

- [Mamba-2](https://arxiv.org/abs/2405.21060)
- [mamba2-minimal](https://github.com/tommyip/mamba2-minimal)
