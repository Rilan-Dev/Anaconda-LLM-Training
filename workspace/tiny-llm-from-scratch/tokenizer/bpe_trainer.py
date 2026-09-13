"""
TinyGPT BPE Trainer

Phase 11

This module trains a Byte Pair Encoding (BPE) tokenizer
from a text corpus and produces:

    tokenizer.json
    vocab.json
    merges.json
    stats.json
    config.json
"""

from pathlib import Path
from collections import Counter
import json
import re
from typing import List, Dict, Tuple


class BPETrainer:
    """
    Byte Pair Encoding Trainer
    """

    def __init__(
        self,
        dataset_path,
        output_dir,
        vocab_size=8000,
        lowercase=True,
    ):

        self.dataset_path = Path(dataset_path)

        self.output_dir = Path(output_dir)

        self.vocab_size = vocab_size

        self.lowercase = lowercase

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # raw documents
        self.documents: List[str] = []

        # cleaned corpus
        self.corpus: List[str] = []

        # word -> frequency
        self.word_freq: Counter = Counter()

        # BPE vocabulary
        self.vocab: Dict[str, int] = {}

        # merge rules
        self.merges: List[Tuple[str, str]] = []

        print("=" * 60)
        print("TinyGPT BPE Trainer")
        print("=" * 60)

        print("Dataset :", self.dataset_path)

        print("Output  :", self.output_dir)

        print("Target Vocabulary :", self.vocab_size)

        print("=" * 60)

    # -------------------------------------------------------

    def load_dataset(self):

        """
        Load merged dataset.
        """

        print("\nLoading dataset...")

        with open(
            self.dataset_path,
            "r",
            encoding="utf-8",
        ) as f:

            dataset = json.load(f)

        self.documents = []

        for sample in dataset:

            instruction = sample.get(
                "instruction",
                ""
            ).strip()

            response = sample.get(
                "response",
                ""
            ).strip()

            if instruction:
                self.documents.append(instruction)

            if response:
                self.documents.append(response)

        print("Documents :", len(self.documents))

        return self.documents

    # -------------------------------------------------------

    def normalize(self):

        """
        Normalize corpus.
        """

        print("\nNormalizing corpus...")

        self.corpus = []

        whitespace = re.compile(r"\s+")

        for text in self.documents:

            if self.lowercase:

                text = text.lower()

            text = whitespace.sub(" ", text)

            text = text.strip()

            if text:

                self.corpus.append(text)

        print("Normalized Documents :", len(self.corpus))

        return self.corpus

    # -------------------------------------------------------

    def build_word_frequency(self):

        """
        Count all words.
        """

        print("\nBuilding word frequencies...")

        self.word_freq = Counter()

        for text in self.corpus:

            self.word_freq.update(
                text.split()
            )

        print("Unique Words :", len(self.word_freq))

        return self.word_freq

    # -------------------------------------------------------

    def build_initial_vocab(self):

        """
        Create initial character vocabulary.

        hello

        becomes

        h e l l o </w>
        """

        print("\nCreating initial vocabulary...")

        self.vocab = {}

        for word, freq in self.word_freq.items():

            chars = " ".join(list(word))

            token = chars + " </w>"

            self.vocab[token] = freq

        print("Vocabulary Entries :", len(self.vocab))

        return self.vocab

        # -------------------------------------------------------

    def get_pair_statistics(self):
        """
        Count the frequency of every adjacent symbol pair.
        """

        print("\nCalculating pair statistics...")

        pair_freq = Counter()

        for word, freq in self.vocab.items():

            symbols = word.split()

            for i in range(len(symbols) - 1):

                pair = (symbols[i], symbols[i + 1])

                pair_freq[pair] += freq

        print("Unique Pairs :", len(pair_freq))

        return pair_freq

        # -------------------------------------------------------

    def find_best_pair(self, pair_freq):
        """
        Return the most frequent pair.
        """

        if len(pair_freq) == 0:
            return None

        best_pair = max(
            pair_freq,
            key=pair_freq.get
        )

        print(
            "Best Pair :",
            best_pair,
            "Frequency:",
            pair_freq[best_pair]
        )

        return best_pair

        # -------------------------------------------------------

    def merge_pair(self, pair):
        """
        Merge one symbol pair throughout the vocabulary.

        Example:

        ('l', 'o')

        becomes

        'lo'
        """

        left, right = pair

        merged = left + right

        new_vocab = {}

        for word, freq in self.vocab.items():

            symbols = word.split()

            new_symbols = []

            i = 0

            while i < len(symbols):

                if (
                    i < len(symbols) - 1
                    and symbols[i] == left
                    and symbols[i + 1] == right
                ):

                    new_symbols.append(merged)

                    i += 2

                else:

                    new_symbols.append(symbols[i])

                    i += 1

            new_word = " ".join(new_symbols)

            new_vocab[new_word] = freq

        self.vocab = new_vocab

        self.merges.append(pair)

        return merged

        # -------------------------------------------------------

    def update_vocabulary(self, iterations=1000):
        """
        Run BPE merge iterations.
        """

        print("\nStarting BPE Training...")

        for step in range(iterations):

            pair_freq = self.get_pair_statistics()

            if len(pair_freq) == 0:

                print("No more pairs.")

                break

            best_pair = self.find_best_pair(pair_freq)

            token = self.merge_pair(best_pair)

            if (step + 1) % 100 == 0:

                print(
                    f"Step {step+1:5d} | "
                    f"Token: {token:15s} | "
                    f"Vocabulary: {len(self.vocab)}"
                )

        print("\nTraining Finished")

        print("Merge Rules :", len(self.merges))