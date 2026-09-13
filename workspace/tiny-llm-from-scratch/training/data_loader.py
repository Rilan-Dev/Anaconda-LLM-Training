"""
TinyLLM DataLoader

Phase 11.06

Responsibilities
----------------
✓ Convert TrainingPair objects to PyTorch tensors
✓ Build Torch Dataset
✓ Create train / validation / test DataLoaders

Author: Mohamed Irsath
Project: TinyLLM From Scratch
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from training.sequence import TrainingPair


# ---------------------------------------------------------
# DataLoader Configuration
# ---------------------------------------------------------

@dataclass
class DataLoaderConfig:
    """
    Configuration for DataLoader.
    """

    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    persistent_workers: bool = False


# ---------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------

class TinyTorchDataset(Dataset):
    """
    PyTorch Dataset wrapping TrainingPair objects.
    """

    def __init__(
        self,
        pairs: List[TrainingPair],
    ):

        self.pairs = pairs

    # -----------------------------------------------------

    def __len__(self):

        return len(self.pairs)

    # -----------------------------------------------------

    def __getitem__(self, index):

        pair = self.pairs[index]

        x = torch.tensor(
            pair.x,
            dtype=torch.long,
        )

        y = torch.tensor(
            pair.y,
            dtype=torch.long,
        )

        return x, y

    # -----------------------------------------------------

    def summary(self):

        print("=" * 60)

        print("TinyTorchDataset")

        print("=" * 60)

        print("Samples :", len(self))

        if len(self):

            x, y = self[0]

            print("Sequence Length :", len(x))

            print("Tensor Type :", x.dtype)

        print("=" * 60)

# ---------------------------------------------------------
# Tiny DataLoader
# ---------------------------------------------------------

class TinyDataLoader:
    """
    Wrapper around torch.utils.data.DataLoader.
    """

    def __init__(
        self,
        dataset: TinyTorchDataset,
        config: Optional[DataLoaderConfig] = None,
    ):

        self.dataset = dataset

        self.config = config or DataLoaderConfig()

    # -----------------------------------------------------

    def create_train_loader(self):

        """
        Create training DataLoader.
        """

        kwargs = dict(
            dataset=self.dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
        )

        if self.config.num_workers > 0:

            kwargs["persistent_workers"] = (
                self.config.persistent_workers
            )

        return DataLoader(**kwargs)

    # -----------------------------------------------------

    def create_validation_loader(self):

        """
        Validation DataLoader.
        """

        kwargs = dict(
            dataset=self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
        )

        if self.config.num_workers > 0:

            kwargs["persistent_workers"] = (
                self.config.persistent_workers
            )

        return DataLoader(**kwargs)

    # -----------------------------------------------------

    def create_test_loader(self):

        """
        Test DataLoader.
        """

        kwargs = dict(
            dataset=self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
        )

        if self.config.num_workers > 0:

            kwargs["persistent_workers"] = (
                self.config.persistent_workers
            )

        return DataLoader(**kwargs)

    # -----------------------------------------------------

    def __len__(self):

        loader = self.create_train_loader()

        return len(loader)

    # -----------------------------------------------------

    @property
    def batch_size(self):

        return self.config.batch_size

    # -----------------------------------------------------

    def summary(self):

        print("=" * 60)

        print("Tiny DataLoader")

        print("=" * 60)

        print("Dataset Samples :", len(self.dataset))

        print("Batch Size      :", self.config.batch_size)

        print("Shuffle         :", self.config.shuffle)

        print("Num Workers     :", self.config.num_workers)

        print("Pin Memory      :", self.config.pin_memory)

        print("Drop Last       :", self.config.drop_last)

        print("Batches         :", len(self))

        print("=" * 60)

    # -----------------------------------------------------

    def first_batch(self):

        """
        Return first batch for debugging.
        """

        loader = self.create_train_loader()

        return next(iter(loader))

# ---------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------

def build_dataset(
    training_pairs: List[TrainingPair],
):
    """
    Build a TinyTorchDataset from training pairs.
    """

    return TinyTorchDataset(training_pairs)


# ---------------------------------------------------------

def build_dataloader(
    training_pairs: List[TrainingPair],
    config: Optional[DataLoaderConfig] = None,
):
    """
    Build a PyTorch DataLoader directly from
    TrainingPair objects.
    """

    dataset = TinyTorchDataset(training_pairs)

    loader = TinyDataLoader(
        dataset=dataset,
        config=config,
    )

    return loader.create_train_loader()


# ---------------------------------------------------------

def inspect_batch(batch):
    """
    Pretty-print a batch.
    """

    x, y = batch

    print("=" * 60)

    print("Batch Inspection")

    print("=" * 60)

    print("Input Shape :", tuple(x.shape))
    print("Target Shape:", tuple(y.shape))

    print()

    print("Input dtype :", x.dtype)
    print("Target dtype:", y.dtype)

    print()

    print("Input Device :", x.device)
    print("Target Device:", y.device)

    print("=" * 60)


# ---------------------------------------------------------

def validate_dataset(dataset: TinyTorchDataset):
    """
    Validate dataset contents.
    """

    print("=" * 60)

    print("Dataset Validation")

    print("=" * 60)

    assert len(dataset) > 0, "Dataset is empty."

    x, y = dataset[0]

    assert isinstance(
        x,
        torch.Tensor,
    )

    assert isinstance(
        y,
        torch.Tensor,
    )

    assert x.dtype == torch.long

    assert y.dtype == torch.long

    assert x.shape == y.shape

    print("✓ Dataset Length :", len(dataset))

    print("✓ Tensor Type :", x.dtype)

    print("✓ Sequence Length :", len(x))

    print("✓ Validation Passed")

    print("=" * 60)


# ---------------------------------------------------------

def validate_loader(loader: DataLoader):
    """
    Validate DataLoader.
    """

    print("=" * 60)

    print("Loader Validation")

    print("=" * 60)

    batch = next(iter(loader))

    x, y = batch

    assert isinstance(x, torch.Tensor)

    assert isinstance(y, torch.Tensor)

    assert x.shape == y.shape

    print("Batch Shape :", tuple(x.shape))

    print("Tensor Type :", x.dtype)

    print("✓ Validation Passed")

    print("=" * 60)


# ---------------------------------------------------------

__all__ = [
    "DataLoaderConfig",
    "TinyTorchDataset",
    "TinyDataLoader",
    "build_dataset",
    "build_dataloader",
    "inspect_batch",
    "validate_dataset",
    "validate_loader",
]