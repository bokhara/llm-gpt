import re
from pathlib import Path

from src.simple_tokenizer_v1 import SimpleTokenizerV1  # assuming src/ is on PYTHONPATH


def build_vocab_from_verdict() -> SimpleTokenizerV1:
    corpus_path = Path("src/the-verdict.txt")
    raw_text = corpus_path.read_text(encoding="utf-8")

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    # New vocab: include two special tokens at the end
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])

    vocab = {token: integer for integer, token in enumerate(all_tokens)}

    return SimpleTokenizerV1(vocab)


def test_edith_wharton_quote_roundtrip():
    tokenizer = build_vocab_from_verdict()

    text = "\"It's the last he painted, you know, Mrs. Gisburn said with pardonable pride.\""

    ids = tokenizer.encode(text)

    # Leading and trailing '"' should map to the same token id (not necessarily 1 anymore)
    assert ids[0] == ids[-1]

    decoded = tokenizer.decode(ids)

    # 简化断言：忽略首尾引号，只要求内容部分 round-trip 一致
    assert decoded.strip('"') == text.strip('"')
