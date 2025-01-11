"""Tokenizer base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional

__all__ = ["Tokenizer"]


class Tokenizer(ABC):
    """Tokenizer base class."""

    oov_token: str
    vocab: dict[int, str | bytes]

    def __init__(self, oov_token: str, vocab: Optional[dict] = None) -> None:
        self.oov_token = oov_token
        self.vocab = vocab or {}

    @property
    def ivocab(self) -> dict[str | bytes, int]:
        """Number of unique tokens."""
        return {token: id for id, token in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens."""
        return len(self.vocab)

    def fit(self, text: str, vocab_size: Optional[int] = None) -> None:
        """Fits the tokenizer to text.

        Parameters
        ----------
        text : str
            Text the tokens should be extracted from.
        vocab_size : int | None, optional
            Number of tokens to be generated. Defaults to ``None``.
        """
        raise NotImplementedError("Fitting not implemented for this tokenizer!")

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encodes text to token ids.

        Parameters
        ----------
        text : str
            Text to be encoded to token ids.

        Returns
        -------
        list[int]
            List of token ids.
        """

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Decodes token ids to text..

        Parameters
        ----------
        token_ids : list[int]
            List of integer token ids to be decoded.

        Returns
        -------
        str
            Decoded text.
        """

    def get_state_dict(self) -> OrderedDict:
        """Returns the tokenizer state dictionary."""
        return OrderedDict(vocab=self.vocab)

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        """Loads the tokenizer state from a state dict.

        Parameters
        ----------
        state_dict : OrderedDict
            State dict containing parameters and buffers.
        """
        for k, v in state_dict.items():
            if k == "ivocab":
                continue
            setattr(self, k, v)
