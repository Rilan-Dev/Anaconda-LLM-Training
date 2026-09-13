"""
TinyGPT BPE Tokenizer

Wrapper around Hugging Face Tokenizers.

Author: TinyLLM Framework
"""

from pathlib import Path
from tokenizers import Tokenizer


class BPETokenizer:

    def __init__(self, tokenizer: Tokenizer):

        self.tokenizer = tokenizer

        self.vocab_size = tokenizer.get_vocab_size()

    # -------------------------------------------------

    @classmethod
    def load(cls, path):

        path = Path(path)

        tokenizer = Tokenizer.from_file(str(path))

        return cls(tokenizer)

    # -------------------------------------------------

    def encode(self, text):

        return self.tokenizer.encode(text).ids

    # -------------------------------------------------

    def decode(self, ids):

        return self.tokenizer.decode(ids)

    # -------------------------------------------------

    def encode_with_tokens(self, text):

        enc = self.tokenizer.encode(text)

        return {

            "ids": enc.ids,

            "tokens": enc.tokens,

        }

    # -------------------------------------------------

    def token_to_id(self, token):

        return self.tokenizer.token_to_id(token)

    # -------------------------------------------------

    def id_to_token(self, idx):

        return self.tokenizer.id_to_token(idx)

    # -------------------------------------------------

    def save(self, path):

        self.tokenizer.save(str(path))

    # -------------------------------------------------

    def __len__(self):

        return self.vocab_size

    # -------------------------------------------------

    def __repr__(self):

        return (

            f"BPETokenizer("

            f"vocab_size={self.vocab_size})"

        )