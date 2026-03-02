from pathlib import Path

import torch

from dataloader_v1 import create_dataloader_v1


def _run_and_print_batch(raw_text: str, batch_size: int, stride: int) -> None:
    """Helper: create dataloader and print first two batches for inspection."""
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=batch_size,
        max_length=4,
        stride=stride,
        shuffle=False,
    )

    it = iter(dataloader)
    inputs1, targets1 = next(it)
    inputs2, targets2 = next(it)

    print(f"\n=== batch_size={batch_size}, stride={stride} ===")
    print("First batch Inputs:\n", inputs1)
    print("First batch Targets:\n", targets1)
    print("Second batch Inputs:\n", inputs2)
    print("Second batch Targets:\n", targets2)

    # 基本关系检查：targets 是 inputs 右移一位+下一个 token
    assert inputs1.shape == targets1.shape
    assert inputs2.shape == targets2.shape
    assert torch.equal(targets1[:, :-1], inputs1[:, 1:])
    assert torch.equal(targets2[:, :-1], inputs2[:, 1:])


def test_gptdataset_v1_various_batch_and_stride():
    """Test GPTDatasetV1 for different batch_size and stride settings.

    We test combinations:
      - batch_size = 4, 8
      - stride = 1, 4
    and print the first two batches for each setting.
    """
    raw_text = Path("src/the-verdict.txt").read_text(encoding="utf-8")

    for batch_size in (4, 8):
        for stride in (1, 4):
            _run_and_print_batch(raw_text, batch_size=batch_size, stride=stride)
