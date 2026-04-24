from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from vrp.constants import DATASET_DIR, OUTPUT_DIR
from vrp.ml.config import TrainConfig
from vrp.ml.ml import VRPAnalysisModel
from vrp.ml.storage import MLTrainingStorage
from vrp.ml.test_model import test_model
from vrp.ml.utils import estimate_pos_weight, run_epoch
from vrp.models.ml_dataset import MLDataset
from vrp.utils.logger import create_logger


def train_model() -> None:
    logger = create_logger("vrp.ml.train_model")
    storage = MLTrainingStorage(OUTPUT_DIR)
    storage.initialize()

    config = TrainConfig()

    train_set = MLDataset(dataset_dir=DATASET_DIR, split="train")
    val_set = MLDataset(dataset_dir=DATASET_DIR, split="val")

    first_sample = train_set[0]
    node_dim = int(first_sample.x.size(-1))
    edge_dim = int(first_sample.edge_attr.size(-1))

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pos_weight_value = estimate_pos_weight(train_set)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = VRPAnalysisModel(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        heads=config.heads,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=config.lr_patience,
    )

    resume_state = storage.load_resume_state(device)

    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        best_val_ap = float(resume_state["best_val_ap"])
        stale_epochs = int(resume_state.get("stale_epochs", 0))
        history = list(resume_state.get("history", []))
        start_epoch = int(resume_state["epoch"]) + 1
        logger.info("Resuming training from epoch %d.", start_epoch)
    else:
        best_val_ap = -float("inf")
        stale_epochs = 0
        history = []
        start_epoch = 1

    logger.info(
        "Training setup: device=%s train_batches=%d val_batches=%d batch_size=%d",
        device,
        len(train_loader),
        len(val_loader),
        config.batch_size,
    )

    last_completed_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, config.epochs + 1):
            logger.info("[epoch %03d/%03d] started", epoch, config.epochs)
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
            )
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            scheduler.step(val_metrics["ap"])

            row = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            last_completed_epoch = epoch

            logger.info(
                f"[epoch {epoch:03d}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_ap={train_metrics['ap']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_ap={val_metrics['ap']:.4f} "
                f"val_auroc={val_metrics['auroc']:.4f}"
            )

            is_best = val_metrics["ap"] > best_val_ap
            if is_best:
                best_val_ap = val_metrics["ap"]
                stale_epochs = 0
            else:
                stale_epochs += 1

            storage.save_training_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch,
                best_val_ap=best_val_ap,
                stale_epochs=stale_epochs,
                history=history,
                node_dim=node_dim,
                edge_dim=edge_dim,
                is_best=is_best,
            )

            if stale_epochs >= config.early_stopping_patience:
                logger.info("Early stopping triggered.")
                break

    except KeyboardInterrupt:
        logger.info(
            "Training interrupted. Progress saved at epoch %d.", last_completed_epoch
        )

        if last_completed_epoch > 0:
            storage.save_training_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=last_completed_epoch,
                best_val_ap=best_val_ap,
                stale_epochs=stale_epochs,
                history=history,
                node_dim=node_dim,
                edge_dim=edge_dim,
                is_best=False,
            )

        return

    logger.info("Best validation AP: %.4f", best_val_ap)

    test_model()


if __name__ == "__main__":
    train_model()
