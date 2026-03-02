import tiktoken
import torch
from torch.utils.data import DataLoader

from simple_tokenizer_v1 import SimpleTokenizerV1
from simple_tokenizer_v2 import SimpleTokenizerV2
from typing import Literal

from torch.utils.data import Dataset


class GPTDatasetV1(Dataset):
    """Dataset that creates (input, target) token sequences for GPT-style training."""

    def __init__(self, txt, tokenizer, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    tokenizer_type: Literal["tiktoken", "simple_v1", "simple_v2"] = "tiktoken",
):
    """Create a DataLoader using GPTDatasetV1.

    tokenizer_type:
      - "tiktoken": use tiktoken GPT-2 BPE tokenizer
      - "simple_v1": use SimpleTokenizerV1 (no <|unk|> handling)
      - "simple_v2": use SimpleTokenizerV2 (with <|unk|>)
    """

    if tokenizer_type == "tiktoken":
        tokenizer = tiktoken.get_encoding("gpt2")
    elif tokenizer_type == "simple_v1":
        raise NotImplementedError("SimpleTokenizerV1 needs a vocab; construct separately")
    elif tokenizer_type == "simple_v2":
        raise NotImplementedError("SimpleTokenizerV2 needs a vocab; construct separately")
    else:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


if __name__ == "__main__":
    # Simple smoke test using the-verdict.txt
    from pathlib import Path

    path = Path("the-verdict.txt")
    raw_text = path.read_text(encoding="utf-8")

    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=1,
        max_length=4,
        stride=1,
        shuffle=False,
    )

    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print("First batch (inputs, targets):")
    print(first_batch)
