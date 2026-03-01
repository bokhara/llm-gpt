import re
from typing import Dict, List


class SimpleTokenizerV1:
    def __init__(self, vocab: Dict[str, int]):
        """Simple whitespace+punctuation tokenizer using a fixed vocab.

        Args:
            vocab: mapping from token string to integer id.
        """
        self.str_to_int: Dict[str, int] = vocab
        self.int_to_str: Dict[int, str] = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        """Encode raw text into a list of token ids.

        Splits on punctuation and whitespace, keeps punctuation as tokens,
        strips and drops empty tokens, then maps each token to its id.
        """
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        ids: List[int] = []
        for s in preprocessed:
            if s not in self.str_to_int:
                raise KeyError(f"Token not in vocab: {s!r}")
            ids.append(self.str_to_int[s])
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token ids back into a text string.

        Joins tokens with spaces, then removes extra spaces before punctuation.
        """
        tokens: List[str] = []
        for i in ids:
            if i not in self.int_to_str:
                raise KeyError(f"ID not in vocab: {i}")
            tokens.append(self.int_to_str[i])

        text = " ".join(tokens)
        text = re.sub(r"\s+([,.?\!\"()\'])", r"\1", text)
        return text


if __name__ == "__main__":
    # small sanity check
    vocab = {"I": 0, "love": 1, "AI": 2, ".": 3}
    tok = SimpleTokenizerV1(vocab)
    ids = tok.encode("I love AI.")
    print("ids:", ids)
    print("decoded:", tok.decode(ids))
