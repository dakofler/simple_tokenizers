"""Word based tokenizer."""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from .base_tokenizer import Tokenizer

WORD_PATTERN = r'([,.:;?_!"()\']|--|\s)'


class WordTokenizer(Tokenizer):
    """Uses whole words as tokens."""

    def __init__(self, oov_token: str = "[unk]") -> None:
        super().__init__(oov_token)

    def fit(self, text: str, vocab_size: Optional[int] = None) -> None:

        # split the text into words
        word_list = re.split(WORD_PATTERN, text)
        word_list = [w for w in word_list if w not in [" ", ""]]

        # get most occuring words
        word_counts = Counter(word_list).most_common((vocab_size or len(word_list)) - 1)
        most_frequent_words = [c[0] for c in word_counts]

        all_tokens = [self.oov_token] + most_frequent_words
        self.vocab = dict(enumerate(all_tokens))

    def encode(self, text: str) -> list[int]:
        # split the text into words
        token_list = re.split(WORD_PATTERN, text)
        token_list = [t for t in token_list if t not in [" ", ""]]
        token_ids = [self.ivocab[t] if t in self.ivocab else 0 for t in token_list]
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        text = " ".join([str(self.vocab[i]) for i in token_ids])
        text = re.sub(r'\s+([,.:?!"()\'])', r"\1", text)  # removes unnecessary spaces
        text = re.sub(r"\n\s", "\n", text)  # removes unnecessary newlines
        return text
