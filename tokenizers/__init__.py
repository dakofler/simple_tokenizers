"""Tokenizers is a collection of tokenization implementations focused on transparency and readability."""

import pathlib

from .base_tokenizer import Tokenizer
from .bpe_tokenizer import BPETokenizer
from .character_tokenizer import CharacterTokenizer
from .word_tokenizer import WordTokenizer

__version__ = pathlib.Path(f"{pathlib.Path(__file__).parent}/VERSION").read_text(
    encoding="utf-8"
)
