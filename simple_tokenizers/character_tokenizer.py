"""Character based tokenizer."""

from __future__ import annotations

import string

from .base_tokenizer import Tokenizer


class CharacterTokenizer(Tokenizer):
    """Uses single characters as tokens. Characters are taken from ``string.printable``."""

    def __init__(self, oov_token: str = "[unk]") -> None:
        all_tokens = [oov_token] + list(string.printable)
        vocab = dict(enumerate(all_tokens))
        super().__init__(oov_token, vocab)

    def encode(self, text: str) -> list[int]:
        token_list = list(text)
        return [self.ivocab[t] if t in self.ivocab else 0 for t in token_list]

    def decode(self, token_ids: list[int]) -> str:
        token_list = [
            self.vocab[i] if i in self.vocab else self.oov_token for i in token_ids
        ]
        return "".join(token_list)
