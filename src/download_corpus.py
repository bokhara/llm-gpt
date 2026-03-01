#!/usr/bin/env python3
"""Download the training text corpus `the-verdict.txt` into the current directory.

Usage:
    python3 download_corpus.py

This script is intentionally simple: you can later expand it to handle
multiple files, retries, checksums, etc.
"""

import pathlib
import sys
import urllib.request

URL = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)
FILENAME = "the-verdict.txt"


def main() -> None:
    target = pathlib.Path(FILENAME)
    if target.exists():
        print(f"File already exists: {target.resolve()}")
        return

    print(f"Downloading {FILENAME} from {URL} ...")
    try:
        urllib.request.urlretrieve(URL, FILENAME)
    except Exception as e:  # noqa: BLE001
        print(f"Download failed: {e}")
        sys.exit(1)

    print(f"Saved to: {target.resolve()}")


if __name__ == "__main__":
    main()
