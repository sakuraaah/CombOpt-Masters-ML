import logging
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
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


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    logger: logging.Logger | None = None,
    epoch: int | None = None,
    phase_name: str | None = None,
    log_every: int | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None

    model.train(is_train)

    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    running_loss = 0.0
    batch_count = 0
    total_batches = len(loader)
    phase_label = phase_name or ("train" if is_train else "eval")
    started_at = time.perf_counter()

    if logger is not None:
        epoch_label = f"{epoch:03d}" if epoch is not None else "???"
        logger.info(
            "[epoch %s] starting %s pass (%d batches)",
            epoch_label,
            phase_label,
            total_batches,
        )

    for batch_idx, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        logits = model(batch)
        target = batch.y.float()
        loss = criterion(logits, target)
        loss_value = float(loss.item())

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        all_logits.append(logits.detach().cpu())
        all_targets.append(target.detach().cpu())
        running_loss += loss_value
        batch_count = batch_idx

        if (
            logger is not None
            and total_batches > 0
            and log_every is not None
            and log_every > 0
            and (batch_idx % log_every == 0 or batch_idx == total_batches)
        ):
            elapsed = time.perf_counter() - started_at
            logger.info(
                "[epoch %s] %s %d/%d (%.1f%%) avg_loss=%.4f elapsed=%.1fs",
                f"{epoch:03d}" if epoch is not None else "???",
                phase_label,
                batch_idx,
                total_batches,
                100.0 * batch_idx / total_batches,
                running_loss / batch_idx,
                elapsed,
            )

    logits = torch.cat(all_logits, dim=0)
    target = torch.cat(all_targets, dim=0)
    target = target.to(torch.int64)
    probs = torch.sigmoid(logits)

    metrics: dict[str, float] = {
        "ap": float(binary_average_precision(probs, target).item()),
    }

    if 0 < int(target.sum()) < int(target.numel()):
        metrics["auroc"] = float(binary_auroc(probs, target).item())
    else:
        metrics["auroc"] = float("nan")

    if batch_count > 0:
        metrics["loss"] = running_loss / batch_count
    else:
        metrics["loss"] = float("nan")

    return metrics
