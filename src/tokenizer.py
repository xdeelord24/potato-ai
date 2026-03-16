"""
Efficient tokenizer for Potato AI.
Uses sentencepiece BPE - compact and fast for CPU inference.
"""

from pathlib import Path
from typing import List, Optional, Union

import sentencepiece as spm


class PotatoTokenizer:
    """
    Wrapper around SentencePiece for efficient tokenization.
    Targets 32K vocab for balance of coverage and embedding size.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        vocab_size: int = 32000,
        pad_token: str = "<|pad|>",
        bos_token: str = "<|bos|>",
        eos_token: str = "<|eos|>",
        unk_token: str = "<|unk|>",
    ):
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        self._sp: Optional[spm.SentencePieceProcessor] = None
        if model_path:
            self.load(model_path)

    def load(self, model_path: Union[str, Path]) -> None:
        """Load a trained SentencePiece model."""
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(str(model_path))
        self.vocab_size = self._sp.get_piece_size()

    def save(self, path: Union[str, Path]) -> None:
        """Save the tokenizer model (requires trained sp)."""
        if self._sp is None:
            raise RuntimeError("No model loaded to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(self._sp.serialized_model_proto())

    def train(
        self,
        files: Union[str, List[str]],
        model_prefix: str = "potato",
        vocab_size: int = 32000,
        model_type: str = "bpe",
        character_coverage: float = 0.9995,
    ) -> None:
        """Train a new SentencePiece model."""
        if isinstance(files, str):
            files = [files]
        spm.SentencePieceTrainer.train(
            input=",".join(files),
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            pad_piece=self.pad_token,
            bos_piece=self.bos_token,
            eos_piece=self.eos_token,
            unk_piece=self.unk_token,
        )
        self.load(f"{model_prefix}.model")
        Path(f"{model_prefix}.model").unlink(missing_ok=True)
        Path(f"{model_prefix}.vocab").unlink(missing_ok=True)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode text to token ids."""
        if self._sp is None:
            raise RuntimeError("Tokenizer not loaded or trained")
        return self._sp.encode(
            text,
            add_bos=add_bos,
            add_eos=add_eos,
            out_type=int,
        )

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token ids to text."""
        if self._sp is None:
            raise RuntimeError("Tokenizer not loaded or trained")
        return self._sp.decode(ids)

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def bos_token_id(self) -> int:
        return 1

    @property
    def eos_token_id(self) -> int:
        return 2

    @property
    def unk_token_id(self) -> int:
        return 3

    def __len__(self) -> int:
        return self.vocab_size if self._sp else 32000


def get_dummy_tokenizer(vocab_size: int = 32000, tmp_dir: Optional[Union[str, Path]] = None) -> PotatoTokenizer:
    """
    Create a minimal tokenizer for testing when no trained model exists.
    """
    import tempfile
    tmp_dir = Path(tmp_dir) if tmp_dir else Path(tempfile.gettempdir()) / "potato_ai"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = tmp_dir / "corpus.txt"
    corpus_path.write_text("The quick brown fox jumps over the lazy dog. Hello world! " * 10)
    model_prefix = str(tmp_dir / "potato_dummy")
    tokenizer = PotatoTokenizer(vocab_size=vocab_size)
    tokenizer.train(
        str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=min(1000, vocab_size),
    )
    return tokenizer
