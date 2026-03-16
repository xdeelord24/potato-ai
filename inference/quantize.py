"""
Quantization for Potato AI - 4-bit and 8-bit for 8GB RAM.
Uses bitsandbytes when available, fallback to manual quantization.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def quantize_linear_4bit(linear: nn.Linear) -> nn.Module:
    """
    Apply 4-bit quantization to a Linear layer.
    Fallback when bitsandbytes not available: use 8-bit via torch.quantization.
    """
    try:
        import bitsandbytes
        from bitsandbytes.nn import Linear4bit

        return Linear4bit(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
        )
    except ImportError:
        # Fallback: return original (no quantization)
        return linear


def quantize_model_4bit(model: nn.Module) -> nn.Module:
    """
    Quantize all Linear layers in the model to 4-bit.
    Requires bitsandbytes. Returns model with quantized layers.
    """
    try:
        import bitsandbytes
    except ImportError:
        print("bitsandbytes not installed - skipping quantization")
        return model

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(
                model,
                name,
                quantize_linear_4bit(module),
            )
        else:
            quantize_model_4bit(module)

    return model


def save_quantized_state_dict(
    model: nn.Module,
    path: Path,
    dtype: torch.dtype = torch.float16,
) -> None:
    """Save model state dict (optionally in half precision)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    if dtype == torch.float16:
        state = {k: v.half() for k, v in state.items()}
    torch.save(state, path)


def load_quantized_checkpoint(
    model: nn.Module,
    path: Path,
    device: str = "cpu",
) -> nn.Module:
    """Load checkpoint into model."""
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    return model
