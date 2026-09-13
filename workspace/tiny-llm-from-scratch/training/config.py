"""
TinyLLM Configuration System

Loads and validates JSON configuration files.

Author : Mohamed Irsath
Project: TinyLLM Framework
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict


# ==========================================================
# Configuration Dataclass
# ==========================================================

@dataclass
class Config:
    """
    TinyLLM configuration object.
    """

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model_name: str

    tokenizer: str

    tokenizer_path: str

    dataset: str

    checkpoint_dir: str

    # --------------------------------------------------
    # Transformer
    # --------------------------------------------------

    block_size: int

    n_layer: int

    n_head: int

    n_embd: int

    dropout: float

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    batch_size: int

    learning_rate: float

    max_steps: int

    eval_interval: int

    eval_iters: int

    device: str

    # --------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "Config":

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        return cls(**payload)

    # --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:

        return self.__dict__.copy()

    # --------------------------------------------------

    def save(self, path: str | Path):

        path = Path(path)

        with open(path, "w", encoding="utf-8") as f:

            json.dump(
                self.to_dict(),
                f,
                indent=4,
            )

    # --------------------------------------------------

    @property
    def parameter_hint(self) -> str:

        if self.n_embd <= 64:
            return "Tiny"

        if self.n_embd <= 128:
            return "Small"

        if self.n_embd <= 256:
            return "Medium"

        return "Large"

    # --------------------------------------------------

    def summary(self):

        print("=" * 60)
        print("TinyLLM Configuration")
        print("=" * 60)

        print(f"Model           : {self.model_name}")
        print(f"Tokenizer       : {self.tokenizer}")
        print(f"Dataset         : {self.dataset}")
        print(f"Block Size      : {self.block_size}")
        print(f"Layers          : {self.n_layer}")
        print(f"Heads           : {self.n_head}")
        print(f"Embedding       : {self.n_embd}")
        print(f"Batch Size      : {self.batch_size}")
        print(f"Learning Rate   : {self.learning_rate}")
        print(f"Max Steps       : {self.max_steps}")
        print(f"Device          : {self.device}")
        print(f"Model Category  : {self.parameter_hint}")

        print("=" * 60)

    # --------------------------------------------------

    def validate(self):

        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        if self.n_layer <= 0:
            raise ValueError("n_layer must be > 0")

        if self.n_head <= 0:
            raise ValueError("n_head must be > 0")

        if self.n_embd <= 0:
            raise ValueError("n_embd must be > 0")

        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be between 0 and 1")

        print("✓ Configuration Valid")