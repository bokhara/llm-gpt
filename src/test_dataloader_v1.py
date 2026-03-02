from pathlib import Path

from dataloader_v1 import create_dataloader_v1


def main() -> None:
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
    print(first_batch)


if __name__ == "__main__":
    main()
