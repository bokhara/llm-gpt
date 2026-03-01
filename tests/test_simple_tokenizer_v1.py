import re
from pathlib import Path

from src.simple_tokenizer_v1 import SimpleTokenizerV1  # assuming src/ is on PYTHONPATH


def build_vocab_from_verdict() -> SimpleTokenizerV1:
    corpus_path = Path("src/the-verdict.txt")
    raw_text = corpus_path.read_text(encoding="utf-8")

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    all_words = sorted(set(preprocessed))
    vocab = {token: integer for integer, token in enumerate(all_words)}

    return SimpleTokenizerV1(vocab)


def test_edith_wharton_quote_roundtrip():
    tokenizer = build_vocab_from_verdict()

    text = "\"It's the last he painted, you know, Mrs. Gisburn said with pardonable pride.\""

    ids = tokenizer.encode(text)

    # Expectation from manual run: leading and trailing '"' map to id 1
    assert ids[0] == 1
    assert ids[-1] == 1

    decoded = tokenizer.decode(ids)
    assert decoded == text
