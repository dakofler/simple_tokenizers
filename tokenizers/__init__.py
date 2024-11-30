"""Tokenizers is a collection of tokenization implementations focused on transparency and readability."""

import pathlib

from .BaseTokenizer import Tokenizer
from .BPETokenizer import BPETokenizer
from .CharacterTokenizer import CharacterTokenizer
from .WordTokenizer import WordTokenizer

__version__ = pathlib.Path(f"{pathlib.Path(__file__).parent}/VERSION").read_text(
    encoding="utf-8"
)
