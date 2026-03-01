import re
from typing import Dict, List


class SimpleTokenizerV2:
    """Simple text tokenizer that handles unknown words.

    - Splits on punctuation and whitespace (same regex as V1)
    - Any token not found in the vocab is mapped to the special token "<|unk|>"
    """

    def __init__(self, vocab: Dict[str, int]):
        """Create a tokenizer with a fixed vocabulary.

        Args:
            vocab: mapping from token string to integer id. Must contain "<|unk|>".
        """
        if "<|unk|>" not in vocab:
            raise ValueError('vocab must contain the "<|unk|>" token')

        self.str_to_int: Dict[str, int] = vocab
        self.int_to_str: Dict[int, str] = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        """Encode raw text into a list of token ids.

        Unknown tokens are mapped to the <|unk|> id.
        """
        # Split on punctuation and whitespace, keep punctuation as tokens
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        # Strip whitespace and drop empty strings
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Map unknown tokens to <|unk|>
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        # Convert to ids
        ids: List[int] = [self.str_to_int[token] for token in preprocessed]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token ids back into text.

        Joins tokens with spaces, then removes extra spaces before punctuation.
        """
        text = " ".join([self.int_to_str[i] for i in ids])
        # Remove spaces before punctuation
        text = re.sub(r'\s+([,.:;?\!"()\'])', r'\1', text)
        return text


if __name__ == "__main__":
    # Small sanity check
    vocab = {
        "<|unk|>": 0,
        "Hello": 1,
        ",": 2,
        "world": 3,
        ".": 4,
    }
    tok = SimpleTokenizerV2(vocab)
    s = "Hello, GPT world!"
    ids = tok.encode(s)
    print("text:", s)
    print("ids:", ids)
    print("decoded:", tok.decode(ids))
