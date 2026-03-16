"""
Data loaders for pre-training, SFT, and agent fine-tuning.
"""

from pathlib import Path
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Simple dataset for causal LM from text files."""

    def __init__(
        self,
        paths: list[Path],
        tokenizer,
        max_length: int = 2048,
        stride: Optional[int] = None,
    ):
        self.paths = paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length // 2
        self._data: list[list[int]] = []

    def _load_and_tokenize(self) -> None:
        for path in self.paths:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            for i in range(0, len(ids) - self.max_length, self.stride):
                self._data.append(ids[i : i + self.max_length])

    def __len__(self) -> int:
        if not self._data:
            self._load_and_tokenize()
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        if not self._data:
            self._load_and_tokenize()
        return {"input_ids": torch.tensor(self._data[idx], dtype=torch.long)}


class InstructionDataset(Dataset):
    """Dataset for instruction tuning (prompt + response)."""

    def __init__(
        self,
        examples: list[dict],
        tokenizer,
        max_length: int = 2048,
        prompt_key: str = "instruction",
        response_key: str = "output",
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_key = prompt_key
        self.response_key = response_key

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        prompt = ex.get(self.prompt_key, "")
        response = ex.get(self.response_key, "")
        full_text = f"{prompt}\n{response}"
        ids = self.tokenizer.encode(full_text, add_bos=True, add_eos=True)
        if len(ids) > self.max_length:
            ids = ids[: self.max_length]
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}


class AgentDataset(Dataset):
    """Dataset for agent fine-tuning (ReAct format)."""

    def __init__(
        self,
        examples: list[dict],
        tokenizer,
        max_length: int = 2048,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        # Format: prompt + thought + tool_call + observation + ...
        text = ex.get("trajectory", ex.get("text", ""))
        ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        if len(ids) > self.max_length:
            ids = ids[: self.max_length]
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}


def get_sample_agent_data() -> list[dict]:
    """Sample agent training data (ReAct format)."""
    return [
        {
            "trajectory": (
                "What is the weather in Tokyo?\n"
                "<thought>I need to check the weather for Tokyo.</thought>\n"
                '<tool_call>get_weather(location="Tokyo")</tool_call>\n'
                "<observation>[Mock weather for Tokyo: 72°F, sunny]</observation>\n"
                "<thought>Based on the observation, the weather in Tokyo is 72°F and sunny.</thought>\n"
                "The weather in Tokyo is 72°F and sunny."
            )
        },
    ]
