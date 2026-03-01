from src.simple_tokenizer_v2 import SimpleTokenizerV2


def test_unknown_tokens_are_mapped_to_unk():
    vocab = {
        "<|unk|>": 0,
        "Hello": 1,
        ",": 2,
        "world": 3,
        ".": 4,
    }
    tok = SimpleTokenizerV2(vocab)

    text = "Hello, GPT world!"
    ids = tok.encode(text)

    # Hello , <|unk|> world <|unk|>
    assert ids == [1, 2, 0, 3, 0]

    decoded = tok.decode(ids)
    # '!' 不在 vocab 中，会被映射成 <|unk|>
    assert decoded == "Hello, <|unk|> world <|unk|>"
