from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from vrp.constants import DATASET_DIR, OUTPUT_DIR
from vrp.ml.storage import MLTrainingStorage
from vrp.ml.train_model import run_epoch
from vrp.ml.utils import estimate_pos_weight
from vrp.models.ml_dataset import MLDataset
from vrp.utils.logger import create_logger


def test_model() -> dict[str, float]:
    logger = create_logger("vrp.ml.test_model")
    storage = MLTrainingStorage(OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = storage.load_model(device=device)

    dataset = MLDataset(dataset_dir=DATASET_DIR, split="test")
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )

    pos_weight_value = estimate_pos_weight(dataset)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    metrics = run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
    )

    storage.save_test_metrics(metrics)
    logger.info("Test metrics saved to %s", storage.test_metrics_path)
    logger.info(
        "test_loss=%.4f test_ap=%.4f test_auroc=%.4f",
        metrics["loss"],
        metrics["ap"],
        metrics["auroc"],
    )
    return metrics


if __name__ == "__main__":
    test_model()
