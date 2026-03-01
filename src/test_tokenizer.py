import re
from pathlib import Path

from simple_tokenizer_v1 import SimpleTokenizerV1


def build_vocab_from_corpus(path: Path):
    raw_text = path.read_text(encoding="utf-8")
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    tokens = [t.strip() for t in tokens if t.strip()]

    # New vocab: include two special tokens at the end
    all_tokens = sorted(list(set(tokens)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])

    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    return vocab


def main() -> None:
    corpus_path = Path("the-verdict.txt")
    if not corpus_path.exists():
        raise SystemExit(f"Corpus file not found: {corpus_path!s}")

    vocab = build_vocab_from_corpus(corpus_path)
    print(f"vocab size: {len(vocab)}")

    tokenizer = SimpleTokenizerV1(vocab)

    text = "It's the last he painted, you know, Mrs. Gisburn said with pardonable pride."
    print("original text:", text)

    ids = tokenizer.encode(text)
    print("encoded ids:", ids)

    decoded = tokenizer.decode(ids)
    print("decoded text:", decoded)


if __name__ == "__main__":
    main()
