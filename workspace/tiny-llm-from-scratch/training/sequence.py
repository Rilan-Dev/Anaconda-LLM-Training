"""
TinyLLM Sequence Builder

Phase 11.05

Responsibilities
----------------
✓ Format instruction/response
✓ Encode samples using tokenizer
✓ Build GPT training sequences
✓ Build datasets
✓ Preview encoded samples
✓ Preview training pairs
✓ Export encoded datasets

Author: Mohamed Irsath
Project: TinyLLM Framework
"""

from __future__ import annotations

import json

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

from training.dataset import Sample


# ==========================================================
# Encoded Sample
# ==========================================================


@dataclass(slots=True)
class EncodedSample:
    """
    One encoded dataset sample.
    """

    text: str
    token_ids: List[int]
    length: int


# ==========================================================
# GPT Training Pair
# ==========================================================


@dataclass(slots=True)
class TrainingPair:
    """
    One autoregressive GPT training pair.

    x -> input tokens

    y -> next-token targets
    """

    x: List[int]
    y: List[int]


# ==========================================================
# Sequence Dataset
# ==========================================================


class SequenceDataset:
    """
    Simple container for GPT training pairs.

    Compatible with future PyTorch Dataset wrapper.
    """

    def __init__(
        self,
        pairs: Optional[List[TrainingPair]] = None,
    ):

        self.pairs = pairs or []

    # ------------------------------------------------------

    def __len__(self):

        return len(self.pairs)

    # ------------------------------------------------------

    def __getitem__(self, index):

        return self.pairs[index]

    # ------------------------------------------------------

    def append(
        self,
        pair: TrainingPair,
    ):

        self.pairs.append(pair)

    # ------------------------------------------------------

    def extend(
        self,
        pairs: Iterable[TrainingPair],
    ):

        self.pairs.extend(pairs)

    # ------------------------------------------------------

    def clear(self):

        self.pairs.clear()

    # ------------------------------------------------------

    def to_list(self):

        return self.pairs


# ==========================================================
# Sequence Builder
# ==========================================================


class SequenceBuilder:
    """
    TinyLLM sequence builder.

    Converts dataset samples into GPT-ready
    training sequences.
    """

    # ------------------------------------------------------

    def __init__(
        self,
        tokenizer,
        template: str = "default",
    ):

        self.tokenizer = tokenizer
        self.template = template

    # ======================================================
    # Prompt Templates
    # ======================================================

    def combine(
        self,
        sample: Sample,
    ) -> str:
        """
        Convert dataset sample into one prompt.
        """

        if self.template == "default":

            return (
                f"User: {sample.instruction}\n\n"
                f"Assistant: {sample.response}"
            )

        if self.template == "chatml":

            return (
                "<|user|>\n"
                f"{sample.instruction}\n"
                "<|assistant|>\n"
                f"{sample.response}"
            )

        if self.template == "alpaca":

            return (
                "### Instruction:\n"
                f"{sample.instruction}\n\n"
                "### Response:\n"
                f"{sample.response}"
            )

        raise ValueError(
            f"Unknown template '{self.template}'"
        )

    # ======================================================
    # Tokenization
    # ======================================================

    def tokenize(
        self,
        text: str,
    ) -> List[int]:
        """
        Encode text using tokenizer.
        """

        if not isinstance(text, str):

            raise TypeError(
                "text must be a string."
            )

        token_ids = self.tokenizer.encode(text)

        if not isinstance(token_ids, list):

            raise TypeError(
                "Tokenizer must return List[int]."
            )

        return token_ids

    # ======================================================
    # Encode Sample
    # ======================================================

    def encode_sample(
        self,
        sample: Sample,
    ) -> EncodedSample:
        """
        Encode one dataset sample.
        """

        text = self.combine(sample)

        token_ids = self.tokenize(text)

        return EncodedSample(
            text=text,
            token_ids=token_ids,
            length=len(token_ids),
        )

    # ======================================================
    # Encode Multiple Samples
    # ======================================================

    def encode(
        self,
        samples: Iterable[Sample],
    ) -> List[EncodedSample]:
        """
        Encode an iterable of samples.
        """

        encoded = []

        for sample in samples:

            encoded.append(
                self.encode_sample(sample)
            )

        return encoded

    # ======================================================
    # Build GPT Training Pairs
    # ======================================================

    def build_training_pairs(
        self,
        token_ids: List[int],
        block_size: int = 128,
        stride: int = 1,
    ) -> List[TrainingPair]:
        """
        Convert a token sequence into GPT next-token prediction pairs.

        Example
        -------
        Tokens:
            [1, 2, 3, 4, 5]

        block_size = 4

        Input:
            [1, 2, 3, 4]

        Target:
            [2, 3, 4, 5]
        """

        if block_size <= 0:
            raise ValueError(
                "block_size must be greater than zero."
            )

        if stride <= 0:
            raise ValueError(
                "stride must be greater than zero."
            )

        if len(token_ids) < block_size + 1:
            return []

        pairs: List[TrainingPair] = []

        max_start = len(token_ids) - block_size - 1

        for start in range(
            0,
            max_start + 1,
            stride,
        ):

            x = token_ids[
                start:start + block_size
            ]

            y = token_ids[
                start + 1:start + block_size + 1
            ]

            pairs.append(
                TrainingPair(
                    x=x,
                    y=y,
                )
            )

        return pairs

    # ======================================================
    # Build Dataset
    # ======================================================

    def build_dataset(
        self,
        encoded_samples: Iterable[EncodedSample],
        block_size: int = 128,
        stride: int = 1,
    ) -> SequenceDataset:
        """
        Build one SequenceDataset from multiple
        encoded samples.
        """

        dataset = SequenceDataset()

        for sample in encoded_samples:

            dataset.extend(

                self.build_training_pairs(
                    token_ids=sample.token_ids,
                    block_size=block_size,
                    stride=stride,
                )

            )

        return dataset

    # ======================================================
    # Statistics
    # ======================================================

    def statistics(
        self,
        encoded_samples: List[EncodedSample],
    ) -> Dict[str, Any]:
        """
        Compute statistics for encoded samples.
        """

        if not encoded_samples:

            return {
                "samples": 0,
                "tokens": 0,
                "min_length": 0,
                "max_length": 0,
                "avg_length": 0.0,
            }

        lengths = [
            sample.length
            for sample in encoded_samples
        ]

        total_tokens = sum(lengths)

        return {

            "samples": len(encoded_samples),

            "tokens": total_tokens,

            "min_length": min(lengths),

            "max_length": max(lengths),

            "avg_length":
                total_tokens / len(encoded_samples),

        }

    # ======================================================
    # Preview Encoded Sample
    # ======================================================

    def preview(
        self,
        encoded: EncodedSample,
        max_tokens: int = 40,
    ) -> None:
        """
        Pretty-print one encoded sample.
        """

        print("=" * 70)

        print("ENCODED SAMPLE")

        print("=" * 70)

        print()

        print(encoded.text)

        print()

        print("-" * 70)

        print("Token IDs")

        print("-" * 70)

        print(
            encoded.token_ids[:max_tokens]
        )

        if encoded.length > max_tokens:

            print("...")

        print()

        print(
            f"Sequence Length : {encoded.length}"
        )

        print("=" * 70)

    # ======================================================
    # Preview Training Pairs
    # ======================================================

    def preview_pairs(
        self,
        pairs: List[TrainingPair],
        count: int = 3,
    ) -> None:
        """
        Display GPT training pairs.
        """

        if not pairs:

            print("No training pairs.")
            return

        print("=" * 70)

        print("GPT TRAINING PAIRS")

        print("=" * 70)

        for index, pair in enumerate(
            pairs[:count],
            start=1,
        ):

            print()

            print(f"Pair {index}")

            print("-" * 70)

            print("Input (x)")

            print(pair.x)

            print()

            print("Target (y)")

            print(pair.y)

            print()

            if hasattr(
                self.tokenizer,
                "decode",
            ):

                try:

                    print("Decoded Input")

                    print(
                        self.tokenizer.decode(
                            pair.x
                        )
                    )

                    print()

                    print("Decoded Target")

                    print(
                        self.tokenizer.decode(
                            pair.y
                        )
                    )

                except Exception:

                    pass

            print("=" * 70)

    # ======================================================
    # Dataset Summary
    # ======================================================

    def dataset_statistics(
        self,
        dataset: SequenceDataset,
    ) -> Dict[str, Any]:
        """
        Compute statistics for a SequenceDataset.
        """

        if len(dataset) == 0:

            return {

                "pairs": 0,

                "block_size": 0,

                "average_length": 0,

            }

        block_size = len(
            dataset[0].x
        )

        return {

            "pairs": len(dataset),

            "block_size": block_size,

            "average_length": block_size,

        }

    # ======================================================
    # Validate Dataset
    # ======================================================

    def validate_dataset(
        self,
        dataset: SequenceDataset,
    ) -> bool:
        """
        Validate dataset consistency.
        """

        if len(dataset) == 0:

            return True

        block_size = len(
            dataset[0].x
        )

        for pair in dataset:

            if len(pair.x) != block_size:

                raise ValueError(
                    "Inconsistent input length."
                )

            if len(pair.y) != block_size:

                raise ValueError(
                    "Inconsistent target length."
                )

        return True

    # ======================================================
    # Export Encoded Samples
    # ======================================================

    def export(
        self,
        encoded_samples: List[EncodedSample],
        output_file: str | Path,
        indent: int = 2,
    ) -> Path:
        """
        Export encoded samples to a JSON file.

        Parameters
        ----------
        encoded_samples:
            List of EncodedSample objects.

        output_file:
            Destination JSON file.

        indent:
            JSON indentation.

        Returns
        -------
        Path
            Path to exported file.
        """

        output_path = Path(output_file)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = [
            asdict(sample)
            for sample in encoded_samples
        ]

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as fp:

            json.dump(
                payload,
                fp,
                indent=indent,
                ensure_ascii=False,
            )

        return output_path

    # ======================================================
    # Load Encoded Samples
    # ======================================================

    def load(
        self,
        input_file: str | Path,
    ) -> List[EncodedSample]:
        """
        Load encoded samples from JSON.
        """

        input_path = Path(input_file)

        if not input_path.exists():

            raise FileNotFoundError(
                input_path
            )

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as fp:

            payload = json.load(fp)

        encoded_samples: List[EncodedSample] = []

        for item in payload:

            encoded_samples.append(

                EncodedSample(

                    text=item["text"],

                    token_ids=item["token_ids"],

                    length=item["length"],

                )

            )

        return encoded_samples

    # ======================================================
    # Export Training Dataset
    # ======================================================

    def export_dataset(
        self,
        dataset: SequenceDataset,
        output_file: str | Path,
        indent: int = 2,
    ) -> Path:
        """
        Export SequenceDataset as JSON.
        """

        output_path = Path(output_file)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = [
            asdict(pair)
            for pair in dataset
        ]

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as fp:

            json.dump(
                payload,
                fp,
                indent=indent,
                ensure_ascii=False,
            )

        return output_path

    # ======================================================
    # Load Training Dataset
    # ======================================================

    def load_dataset(
        self,
        input_file: str | Path,
    ) -> SequenceDataset:
        """
        Load a previously exported SequenceDataset.
        """

        input_path = Path(input_file)

        if not input_path.exists():

            raise FileNotFoundError(
                input_path
            )

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as fp:

            payload = json.load(fp)

        dataset = SequenceDataset()

        for item in payload:

            dataset.append(

                TrainingPair(

                    x=item["x"],

                    y=item["y"],

                )

            )

        return dataset

    # ======================================================
    # Decode Tokens
    # ======================================================

    def decode(
        self,
        token_ids: List[int],
    ) -> str:
        """
        Decode token IDs using tokenizer.
        """

        if not hasattr(
            self.tokenizer,
            "decode",
        ):

            raise AttributeError(
                "Tokenizer does not implement decode()."
            )

        return self.tokenizer.decode(
            token_ids
        )

    # ======================================================
    # Batch Encode
    # ======================================================

    def batch_encode(
        self,
        samples: Iterable[Sample],
    ) -> List[EncodedSample]:
        """
        Alias for encode().

        Useful for readability.
        """

        return self.encode(samples)

    # ======================================================
    # Batch Decode
    # ======================================================

    def batch_decode(
        self,
        sequences: Iterable[List[int]],
    ) -> List[str]:
        """
        Decode multiple token sequences.
        """

        return [

            self.decode(tokens)

            for tokens in sequences

        ]

    # ======================================================
    # Summary
    # ======================================================

    def summary(
        self,
    ) -> None:
        """
        Print configuration summary.
        """

        print("=" * 70)

        print("TinyLLM Sequence Builder")

        print("=" * 70)

        print(
            f"Template  : {self.template}"
        )

        print(
            f"Tokenizer : {self.tokenizer.__class__.__name__}"
        )

        print("=" * 70)

    # ======================================================
    # Representation
    # ======================================================

    def __repr__(
        self,
    ) -> str:

        return (

            f"{self.__class__.__name__}("
            f"template='{self.template}', "
            f"tokenizer={self.tokenizer.__class__.__name__}"
            f")"

        )

    # ======================================================
    # String Representation
    # ======================================================

    def __str__(
        self,
    ) -> str:

        return self.__repr__()