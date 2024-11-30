"""Word based tokenizer."""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from .BaseTokenizer import Tokenizer

WORD_PATTERN = r'([,.:;?_!"()\']|--|\s)'


class WordTokenizer(Tokenizer):
    """Uses whole words as tokens."""

    def __init__(self, oov_token: str = "<|unk|>") -> None:
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
        self.ivocab = {t: i for i, t in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        word_list = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        word_list = [w for w in word_list if w not in [" ", ""]]
        return [self.ivocab[s] if s in self.ivocab else 0 for s in word_list]

    def decode(self, token_ids: list[int]) -> str:
        text = " ".join([self.vocab[i] for i in token_ids])
        text = re.sub(r'\s+([,.:?!"()\'])', r"\1", text)
        return re.sub(r"\n\s", "\n", text)
