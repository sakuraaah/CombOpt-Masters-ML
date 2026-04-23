import torch
from torch.utils.data import Dataset
from torchmetrics.functional.classification import (
    binary_average_precision,
    binary_auroc,
)


def estimate_pos_weight(
    dataset: Dataset,
    max_samples: int | None = None,
    cap: float = 50.0,
) -> float:
    pos = 0.0
    total = 0.0
    limit = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    for i in range(limit):
        y = dataset[i].y
        pos += float(y.sum().item())
        total += float(y.numel())

    if pos == 0:
        return 1.0

    neg = total - pos
    value = neg / pos
    return min(value, cap)


def compute_threshold_metrics(
    probs: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    preds = (probs >= threshold).to(torch.int64)
    target = target.to(torch.int64)

    tp = int(((preds == 1) & (target == 1)).sum())
    fp = int(((preds == 1) & (target == 0)).sum())
    fn = int(((preds == 0) & (target == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)

    return {
        f"precision@{threshold:.2f}": precision,
        f"recall@{threshold:.2f}": recall,
        f"f1@{threshold:.2f}": f1,
    }


def summarize_epoch_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    total_loss: float,
    total_edges: int,
) -> dict[str, float]:
    target = target.to(torch.int64)
    probs = torch.sigmoid(logits)

    metrics: dict[str, float] = {
        "loss": total_loss / max(total_edges, 1),
        "ap": float(binary_average_precision(probs, target).item()),
    }

    if 0 < int(target.sum()) < int(target.numel()):
        metrics["auroc"] = float(binary_auroc(probs, target).item())
    else:
        metrics["auroc"] = float("nan")

    metrics.update(compute_threshold_metrics(probs, target, threshold=0.5))
    return metrics
