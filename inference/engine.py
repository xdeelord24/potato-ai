"""
CPU-optimized inference engine for Potato AI.
Uses PyTorch with optional torch.compile for faster inference.
"""

from pathlib import Path
from typing import List, Optional

import torch

from src.ssm import PotatoConfig, PotatoLM


class InferenceEngine:
    """
    Inference engine for Potato LM - optimized for CPU, 8GB RAM.
    """

    def __init__(
        self,
        model: Optional[PotatoLM] = None,
        config: Optional[PotatoConfig] = None,
        checkpoint_path: Optional[Path] = None,
        device: str = "cpu",
        use_compile: bool = False,
    ):
        if model is not None:
            self.model = model
            self.config = model.config
        elif checkpoint_path is not None and config is not None:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            # Infer vocab_size from checkpoint
            if "embedding.weight" in state:
                config.vocab_size = state["embedding.weight"].shape[0]
            self.config = config
            self.model = PotatoLM(config)
            self.model.load_state_dict(state, strict=False)
        else:
            raise ValueError("Provide either model or (checkpoint_path + config)")

        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

        if use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """Generate tokens autoregressively."""
        input_ids = input_ids.to(self.device)
        return self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id or self.config.eos_token_id,
            pad_token_id=pad_token_id or self.config.pad_token_id,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        cache: Optional[List] = None,
    ) -> tuple[torch.Tensor, List]:
        """Single forward pass."""
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            return self.model(input_ids, cache)

    @torch.no_grad()
    def complete(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 64,
        temperature: float = 0.8,
    ) -> List[int]:
        """Complete a prompt, returning token ids."""
        input_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=self.device,
        )
        generated = self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return generated[0].tolist()
