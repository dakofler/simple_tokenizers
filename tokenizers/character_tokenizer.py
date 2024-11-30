"""Character based tokenizer."""

from __future__ import annotations

import string

from .base_tokenizer import Tokenizer


class CharacterTokenizer(Tokenizer):
    """Uses single characters as tokens. Characters are taken from ``string.printable``."""

    def __init__(self, oov_token: str = "<|unk|>") -> None:
        all_tokens = [oov_token, "\n"] + list(string.printable)
        vocab = dict(enumerate(all_tokens))  # id -> token
        ivocab = {token: token_id for token_id, token in vocab.items()}  # token -> id
        super().__init__(oov_token, vocab, ivocab)

    def encode(self, text: str) -> list[int]:
        token_list = list(text)
        return [self.ivocab[t] if t in self.ivocab else 0 for t in token_list]

    def decode(self, token_ids: list[int]) -> str:
        unk_token = self.vocab[0]
        token_list = [
            self.vocab[i] if i in self.vocab else unk_token for i in token_ids
        ]
        return "".join(token_list)
