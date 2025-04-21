"""Simple tokenizers is a collection of tokenization implementations focused on transparency and readability."""

import pathlib

from .base_tokenizer import *
from .bpe_tokenizer import *
from .character_tokenizer import *
from .word_tokenizer import *

__version__ = pathlib.Path(f"{pathlib.Path(__file__).parent}/VERSION").read_text(
    encoding="utf-8"
)
