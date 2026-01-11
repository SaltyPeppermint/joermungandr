import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, final

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


@dataclass
class RerankerExample:
    """A single reranker training example.

    Attributes:
        start: Start state/sentence.
        goal: Goal state/sentence.
        guide: Guide/middle sentence to evaluate.
        label: 1 = valid guide, 0 = invalid guide.
    """

    start: str
    goal: str
    guide: str
    label: int


# Prompt template following Qwen3-Reranker format
SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
)

DEFAULT_INSTRUCTION = (
    "Given a start state and goal state, determine if the guide is a valid intermediate step."
)


def format_reranker_input(
    start: str,
    goal: str,
    guide: str,
    instruction: str | None = None,
) -> str:
    """Format (start, goal, guide) triple for reranker scoring.

    Maps to reranker's expected format:
    - Query: "Start: {start} Goal: {goal}"
    - Document: "{guide}"

    Args:
        start: Start state/sentence.
        goal: Goal state/sentence.
        guide: Guide/middle sentence.
        instruction: Optional custom instruction.

    Returns:
        Formatted string for the reranker.
    """
    if instruction is None:
        instruction = DEFAULT_INSTRUCTION

    query = f"Start: {start} Goal: {goal}"
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {guide}"


def format_messages(
    start: str,
    goal: str,
    guide: str,
    instruction: str | None = None,
) -> list[dict[str, str]]:
    """Format as chat messages for tokenizer.apply_chat_template().

    Args:
        start: Start state/sentence.
        goal: Goal state/sentence.
        guide: Guide/middle sentence.
        instruction: Optional custom instruction.

    Returns:
        List of message dicts for chat template.
    """
    user_content = format_reranker_input(start, goal, guide, instruction)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def load_jsonl(path: str | Path) -> Iterator[RerankerExample]:
    """Load examples from a JSONL file.

    Expected format per line:
        {"start": "...", "goal": "...", "guide": "...", "label": 0 or 1}

    Args:
        path: Path to the JSONL file.

    Yields:
        RerankerExample instances.
    """
    with open(path) as f:
        for line in f:
            data: dict[str, Any] = json.loads(line)
            yield RerankerExample(
                start=data["start"], goal=data["goal"], guide=data["guide"], label=data["label"]
            )


@final
class RerankerDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch Dataset for reranker fine-tuning."""

    examples: list[RerankerExample]
    tokenizer: PreTrainedTokenizer
    max_length: int
    instruction: str | None
    suffix: str

    def __init__(
        self,
        examples: list[RerankerExample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 8192,
        instruction: str | None = None,
    ):
        """Initialize the dataset.

        Args:
            examples: List of RerankerExample instances.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.
            instruction: Optional custom instruction for all examples.
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction

        # Precompute the suffix tokens (assistant turn with thinking tags)
        self.suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]

        messages = format_messages(example.start, example.goal, example.guide, self.instruction)

        # Apply chat template and add assistant prefix
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        assert isinstance(text, str), "Expected string from apply_chat_template with tokenize=False"
        text = text + self.suffix

        # Tokenize
        encoded = self.tokenizer(
            text, truncation=True, max_length=self.max_length, padding=False, return_tensors=None
        )

        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(example.label, dtype=torch.long),
        }


def load_reranker_dataset(
    path: str | Path,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 8192,
    instruction: str | None = None,
) -> RerankerDataset:
    """Load a RerankerDataset from a JSONL file.

    Args:
        path: Path to the JSONL file.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
        instruction: Optional custom instruction.

    Returns:
        RerankerDataset instance.
    """
    examples = list(load_jsonl(path))
    return RerankerDataset(examples, tokenizer, max_length, instruction)


def collate_reranker_batch(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Collate function for DataLoader with left-padding.

    Args:
        batch: List of encoded examples.
        pad_token_id: Token ID to use for padding.

    Returns:
        Batched and padded tensors.
    """
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids: list[torch.Tensor] = []
    attention_mask: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        # Left-pad input_ids and attention_mask
        input_ids.append(
            torch.cat([torch.full((pad_len,), pad_token_id, dtype=torch.long), item["input_ids"]])
        )
        attention_mask.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long), item["attention_mask"]])
        )
        labels.append(item["labels"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }
