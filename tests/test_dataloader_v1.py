from pathlib import Path

import torch

from dataloader_v1 import create_dataloader_v1


def test_gptdataset_v1_shift_relation():
    """Verify that targets are inputs shifted by one token for a small config.

    We use a tiny max_length and stride so it's easy to inspect the first batch.
    """
    raw_text = Path("src/the-verdict.txt").read_text(encoding="utf-8")

    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=4,
        max_length=4,
        stride=1,
        shuffle=False,
    )

    inputs, targets = next(iter(dataloader))

    print("Inputs batch:\n", inputs)
    print("Targets batch:\n", targets)

    # 检查形状一致
    assert inputs.shape == targets.shape

    # 对除了最后一个位置外，targets 应该等于 inputs 向右平移 1 位
    # targets[:, :-1] == inputs[:, 1:]
    assert torch.equal(targets[:, :-1], inputs[:, 1:])

    # 最后一列应该等于“下一个 token”，即整段序列向前再看一步
    # 这里只做一个简单检查：至少有一个样本在最后一位与前一行的首位不同，
    # 以确保不是简单拷贝（不严格，但足够防止明显错误）。
    assert not torch.equal(targets[:, -1], inputs[:, -1])
