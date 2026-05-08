"""Evaluate the trained model on the validation set and save metrics to app/static/metrics.json.

Computes:
  - Overall accuracy, macro F1, weighted F1
  - Per-class precision, recall, F1, AUC-ROC (one-vs-rest)
  - 8×8 confusion matrix
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from torchvision.models import efficientnet_b0
from tqdm import tqdm

from dataset import build_loaders


@torch.no_grad()
def collect_predictions(model: nn.Module, loader, device: str):
    model.eval()
    all_labels, all_probs = [], []
    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        probs = F.softmax(model(images), dim=1).cpu().numpy()
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)
    return np.array(all_labels), np.array(all_probs)


def load_model(model_path: Path) -> tuple[nn.Module, list[str]]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, len(classes)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, classes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--model-path", default="../models/best_model.pt")
    parser.add_argument("--output", default="../app/static/metrics.json")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    model, classes = load_model(Path(args.model_path))
    model.to(device)

    _, val_loader, _ = build_loaders(
        Path(args.data_dir), batch_size=32, num_workers=args.num_workers
    )

    labels, probs = collect_predictions(model, val_loader, device)
    preds = probs.argmax(axis=1)

    report = classification_report(
        labels, preds, target_names=classes, output_dict=True
    )

    cm = confusion_matrix(labels, preds).tolist()

    labels_bin = label_binarize(labels, classes=list(range(len(classes))))
    auc_per_class = {
        cls: float(roc_auc_score(labels_bin[:, i], probs[:, i]))
        for i, cls in enumerate(classes)
    }

    metrics = {
        "accuracy":    round(report["accuracy"], 4),
        "macro_f1":    round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "classes": classes,
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"], 4),
                "f1":        round(report[cls]["f1-score"], 4),
                "auc":       round(auc_per_class[cls], 4),
                "support":   int(report[cls]["support"]),
            }
            for cls in classes
        },
        "confusion_matrix": cm,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved → {out}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("\nPer-class AUC-ROC:")
    for cls, auc in auc_per_class.items():
        print(f"  {auc:.4f}  {cls}")


if __name__ == "__main__":
    main()
