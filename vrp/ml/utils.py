import torch
from torch.utils.data import Dataset
from torchmetrics.functional.classification import (
    binary_average_precision,
    binary_auroc,
)


def estimate_pos_weight(dataset: Dataset) -> float:
    pos = 0.0
    total = 0.0

    for i in range(len(dataset)):
        y = dataset[i].y
        pos += float(y.sum().item())
        total += float(y.numel())

    if pos == 0:
        return 1.0

    neg = total - pos

    return neg / pos


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

    return metrics
