"""
TinyLLM Dataset Module

Responsible for:

✓ Loading datasets
✓ Validating samples
✓ Dataset statistics
✓ Search
✓ Export

Author : Mohamed Irsath
Project : TinyLLM Framework
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Iterator


# ==========================================================
# Dataset Sample
# ==========================================================

@dataclass
class Sample:

    instruction: str

    response: str


# ==========================================================
# Tiny Dataset
# ==========================================================

class TinyDataset:

    def __init__(self, samples: List[Sample]):

        self.samples = samples

    # -----------------------------------------------------

    @classmethod
    def load(cls, path):

        path = Path(path)

        if not path.exists():

            raise FileNotFoundError(path)

        with open(path, "r", encoding="utf-8") as f:

            payload = json.load(f)

        samples = []

        for item in payload:

            instruction = (
                item.get("instruction")
                or item.get("user")
                or ""
            ).strip()
            
            response = (
                item.get("response")
                or item.get("assistant")
                or ""
            ).strip()

            if instruction and response:

                samples.append(

                    Sample(
                        instruction=instruction,
                        response=response,
                    )
                )

        return cls(samples)

    # -----------------------------------------------------

    def __len__(self):

        return len(self.samples)

    # -----------------------------------------------------

    def __getitem__(self, index):

        return self.samples[index]

    # -----------------------------------------------------

    def __iter__(self) -> Iterator[Sample]:

        return iter(self.samples)

    # -----------------------------------------------------

    def statistics(self):

        total_instruction = sum(

            len(x.instruction)

            for x in self.samples

        )

        total_response = sum(

            len(x.response)

            for x in self.samples

        )

        return {

            "samples": len(self),

            "avg_instruction":

                total_instruction / len(self),

            "avg_response":

                total_response / len(self),

        }

    # -----------------------------------------------------

    def summary(self):

        stats = self.statistics()

        print("=" * 60)

        print("TinyLLM Dataset")

        print("=" * 60)

        print("Samples :", stats["samples"])

        print(

            "Avg Instruction :",

            round(stats["avg_instruction"], 2)

        )

        print(

            "Avg Response :",

            round(stats["avg_response"], 2)

        )

        print("=" * 60)

    # -----------------------------------------------------

    def search(self, keyword):

        keyword = keyword.lower()

        results = []

        for sample in self.samples:

            if (

                keyword in sample.instruction.lower()

                or

                keyword in sample.response.lower()

            ):

                results.append(sample)

        return results

    # -----------------------------------------------------

    def head(self, n=5):

        return self.samples[:n]

    # -----------------------------------------------------

    def tail(self, n=5):

        return self.samples[-n:]

    # -----------------------------------------------------

    def export_json(self, path):

        path = Path(path)

        payload = []

        for sample in self.samples:

            payload.append({

                "instruction": sample.instruction,

                "response": sample.response,

            })

        with open(path, "w", encoding="utf-8") as f:

            json.dump(

                payload,

                f,

                indent=4,

                ensure_ascii=False,

            )

    # -----------------------------------------------------

    def validate(self):

        errors = []

        for idx, sample in enumerate(self.samples):

            if not sample.instruction:

                errors.append(

                    f"Row {idx}: Empty instruction"

                )

            if not sample.response:

                errors.append(

                    f"Row {idx}: Empty response"

                )

        return errors