"""Configuration for the Potato AI model - optimized for 8GB RAM systems."""

from dataclasses import dataclass


@dataclass
class PotatoConfig:
    """Model configuration targeting 8GB RAM, CPU-only inference."""

    # Core dimensions
    d_model: int = 768  # Hidden dimension (768 for ~1B params)
    n_layer: int = 24  # Number of SSM blocks
    vocab_size: int = 32000  # Vocabulary size

    # Mamba-2 / SSM parameters
    d_state: int = 128  # SSM state dimension
    d_conv: int = 4  # Convolution kernel size
    expand: int = 2  # Expansion factor (d_inner = expand * d_model)
    headdim: int = 64  # Head dimension for SSD
    chunk_size: int = 64  # Matrix partition size for SSD

    # Gated FFN (SwiGLU)
    ffn_mult: int = 4  # FFN hidden = ffn_mult * d_model
    ffn_dropout: float = 0.0

    # Context
    max_seq_len: int = 4096  # Max context length

    # Misc
    pad_vocab_size_multiple: int = 16
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0, "d_inner must be divisible by headdim"
        self.nheads = self.d_inner // self.headdim

        # Pad vocab for efficient matmul
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )

    @classmethod
    def tiny(cls) -> "PotatoConfig":
        """~10M params for validation and testing."""
        return cls(
            d_model=256,
            n_layer=4,
            vocab_size=32000,
            d_state=64,
            expand=2,
            headdim=32,
            chunk_size=32,
            ffn_mult=2,
        )

    @classmethod
    def small(cls) -> "PotatoConfig":
        """~350M params - good balance for potato computers."""
        return cls(
            d_model=512,
            n_layer=16,
            vocab_size=32000,
            d_state=96,
            expand=2,
            headdim=64,
        )

    @classmethod
    def base(cls) -> "PotatoConfig":
        """~1B params - full target for 8GB RAM."""
        return cls()
