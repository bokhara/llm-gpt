#!/usr/bin/env python3
"""Tokenize Edith Wharton short story `the-verdict.txt` using a simple regex tokenizer.

Steps:
- Read raw text from the-verdict.txt
    preprocessed = re.split(r"([,.:;?_!\"()']|--|\s)", raw_text)
- Strip and drop empty tokens
- Print token count and first 30 tokens for inspection
"""

import re
from pathlib import Path

CORPUS_FILE = Path("the-verdict.txt")


def main() -> None:
    if not CORPUS_FILE.exists():
        raise SystemExit(f"Corpus file not found: {CORPUS_FILE!s}")

    raw_text = CORPUS_FILE.read_text(encoding="utf-8")

    # Tokenize: keep punctuation as separate tokens, split on whitespace
    preprocessed = re.split(r"([,.:;?_!\"()']|--|\s)", raw_text)
    # Strip whitespace and drop empty strings
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    print(f"Number of tokens (without pure whitespace): {len(preprocessed)}")
    print("First 30 tokens:")
    print(preprocessed[:30])


if __name__ == "__main__":
    main()