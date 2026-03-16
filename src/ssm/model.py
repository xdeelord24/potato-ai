"""Full Potato LM: embedding + SSM layers + LM head."""

from typing import List, Optional

import torch
from torch import nn

from .config import PotatoConfig
from .mamba2_block import InferenceCache, Mamba2Block, RMSNorm


class PotatoLM(nn.Module):
    """
    Potato AI language model: SSM backbone + gated FFN, optimized for 8GB RAM.
    """

    def __init__(self, config: PotatoConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([Mamba2Block(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.LongTensor,
        cache: Optional[List[Optional[InferenceCache]]] = None,
    ) -> tuple[torch.Tensor, List[InferenceCache]]:
        """
        Args:
            input_ids: (batch, seqlen) token ids
            cache: list of per-layer caches for step mode (seqlen=1)

        Returns:
            logits: (batch, seqlen, vocab_size)
            cache: updated per-layer caches
        """
        seqlen = input_ids.shape[1]
        if cache is None:
            cache = [None for _ in range(self.config.n_layer)]

        x = self.embedding(input_ids)
        new_cache: List[InferenceCache] = []

        for i, layer in enumerate(self.layers):
            x, h = layer(x, cache[i])
            new_cache.append(h)

        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits[:, :seqlen], new_cache

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Generate tokens autoregressively using constant-time per-token inference.
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Process prefix in chunks
        prefix = input_ids
        chunk_size = self.config.chunk_size
        n_chunked = (prefix.shape[1] // chunk_size) * chunk_size

        if n_chunked > 0:
            prefix_chunk = prefix[:, :n_chunked]
            _, cache = self(prefix_chunk, None)
        else:
            cache = [
                InferenceCache.alloc(batch_size, self.config, device)
                for _ in range(self.config.n_layer)
            ]

        # Process remaining prefix tokens one by one
        for i in range(n_chunked, prefix.shape[1]):
            tok = prefix[:, i : i + 1]
            _, cache = self(tok, cache)

        # Generate
        generated = prefix
        for _ in range(max_new_tokens - 1):
            next_tok = generated[:, -1:]
            logits, cache = self(next_tok, cache)
            logits = logits[:, -1]

            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumsum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumsum - (cumsum > top_p).float() > 0
                sorted_logits[mask] = -float("inf")
                logits = torch.zeros_like(logits).scatter_(
                    1, sorted_idx, sorted_logits
                )

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return generated

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
