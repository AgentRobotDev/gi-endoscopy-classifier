"""Kvasir-v2 dataset loaders with train/val split."""

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_loaders(
    data_dir: Path,
    batch_size: int,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[str]]:
    root = data_dir / "kvasir-v2"
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset not found at {root}. Run: python download_dataset.py"
        )

    train_ds = datasets.ImageFolder(root, transform=TRAIN_TRANSFORM)
    val_ds = datasets.ImageFolder(root, transform=VAL_TRANSFORM)

    n = len(train_ds)
    n_val = int(n * val_split)
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    train_subset = Subset(train_ds, indices[n_val:])
    val_subset = Subset(val_ds, indices[:n_val])

    # macOS + MPS can deadlock with multiple workers
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, train_ds.classes
