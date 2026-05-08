"""Train EfficientNet-B0 on Kvasir-v2.

Two-phase training:
  Phase 1 (epochs 1 → unfreeze_epoch): classifier head only, backbone frozen.
  Phase 2 (unfreeze_epoch → end):      full model, lower LR.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import build_loaders
from model import build_model


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, leave=False, desc="  train"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, leave=False, desc="  val  "):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--output-dir", default="../models")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--unfreeze-epoch", type=int, default=5,
                        help="Epoch at which to unfreeze backbone (0 = never freeze)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (use 0 on macOS/MPS to avoid deadlocks)")
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    train_loader, val_loader, classes = build_loaders(
        data_dir, args.batch_size, args.val_split, args.num_workers
    )
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    freeze = args.unfreeze_epoch > 0
    model = build_model(num_classes=len(classes), freeze_backbone=freeze).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=min(args.unfreeze_epoch or args.epochs, args.epochs))

    best_val_acc = 0.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        if args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch:
            print(f"\nEpoch {epoch}: unfreezing backbone, dropping LR to {args.lr / 10:.2e}")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = Adam(model.parameters(), lr=args.lr / 10)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch + 1)

        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = dict(epoch=epoch, train_loss=round(train_loss, 4), train_acc=round(train_acc, 4),
                   val_loss=round(val_loss, 4), val_acc=round(val_acc, 4))
        history.append(row)
        print(f"  train loss={train_loss:.4f} acc={train_acc:.3f} | val loss={val_loss:.4f} acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "classes": classes, "val_acc": val_acc},
                output_dir / "best_model.pt",
            )
            print(f"  ✓ New best — saved (val_acc={val_acc:.3f})")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val_acc: {best_val_acc:.3f}")
    print(f"Model saved to {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
