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

# Prepare data (trains SentencePiece tokenizer on data/*.txt)
python scripts/prepare_data.py

# Pre-train
python -m training.pretrain --config tiny --max_steps 500

# Run inference
python scripts/run_inference.py --checkpoint checkpoints/pretrain_tiny_final.pt --prompt "Machine learning is"
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

1. **Prepare data**: Add `.txt` files to `data/`, then `python scripts/prepare_data.py` to train tokenizer
2. **Pre-train**: `python -m training.pretrain --config tiny --max_steps 1000`
3. **SFT**: `python -m training.sft --checkpoint checkpoints/pretrain_tiny_final.pt`
4. **Agent**: `python -m training.agent_finetune --checkpoint checkpoints/sft_tiny.pt`

## Data

- `data/sample.txt`, `prose.txt`, `facts.txt`, `code.txt` - diverse pre-training text
- `data/instructions.jsonl` - instruction tuning (instruction, output)
- `data/agent_trajectories.jsonl` - ReAct agent examples
- Run `python scripts/prepare_data.py --download_wikitext` for more data

## Configs

- `tiny`: ~11M params (validation)
- `small`: ~350M params
- `base`: ~1B params (8GB target)

## References

- [Mamba-2](https://arxiv.org/abs/2405.21060)
- [mamba2-minimal](https://github.com/tommyip/mamba2-minimal)
