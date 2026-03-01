import tiktoken


class TestTikTokenEncoding:
    def test_endoftext_special_token_roundtrip(self) -> None:
        enc = tiktoken.get_encoding("gpt2")

        text = (
            "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
            " of someunknownPlace."
        )

        integers = enc.encode(text, allowed_special={"<|endoftext|>"})
        # 打印出来方便本地观察（pytest 会捕获输出）
        print("text:", text)
        print("token ids:", integers)

        decoded = enc.decode(integers)
        # round-trip 应该能还原原始文本
        assert decoded == text
        # 确认特殊 token 没被拆掉
        assert "<|endoftext|>" in decoded
