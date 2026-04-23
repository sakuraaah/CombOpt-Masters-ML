from pathlib import Path
import json
from dataclasses import asdict

import torch

from vrp.ml.config import TrainConfig
from vrp.ml.ml import VRPAnalysisModel


class MLTrainingStorage:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.best_checkpoint_path = output_dir / "best.pt"
        self.latest_checkpoint_path = output_dir / "latest.pt"
        self.history_path = output_dir / "history.json"
        self.progress_path = output_dir / "progress.json"
        self.test_metrics_path = output_dir / "test_metrics.json"

    def initialize(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_training_state(
        self,
        model: VRPAnalysisModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        config: TrainConfig,
        epoch: int,
        best_val_ap: float,
        stale_epochs: int,
        history: list[dict[str, float | int]],
        node_dim: int,
        edge_dim: int,
        is_best: bool,
    ) -> None:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "best_val_ap": best_val_ap,
            "stale_epochs": stale_epochs,
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "history": history,
        }

        torch.save(checkpoint, self.latest_checkpoint_path)
        if is_best:
            torch.save(checkpoint, self.best_checkpoint_path)

        self.history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        self.progress_path.write_text(
            json.dumps(
                {
                    "epoch": epoch,
                    "best_val_ap": best_val_ap,
                    "stale_epochs": stale_epochs,
                    "history_length": len(history),
                    "has_best_checkpoint": self.best_checkpoint_path.exists(),
                    "has_latest_checkpoint": self.latest_checkpoint_path.exists(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def load_checkpoint(
        self,
        device: torch.device,
        checkpoint_path: Path | None = None,
    ) -> dict[str, object]:
        resolved_path = checkpoint_path or self.best_checkpoint_path
        if not resolved_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")

        return torch.load(resolved_path, map_location=device, weights_only=False)

    def load_model(
        self,
        device: torch.device,
        checkpoint_path: Path | None = None,
    ) -> tuple[VRPAnalysisModel, TrainConfig]:
        checkpoint = self.load_checkpoint(device=device, checkpoint_path=checkpoint_path)
        config = TrainConfig(**checkpoint["config"])

        model = VRPAnalysisModel(
            node_dim=checkpoint["node_dim"],
            edge_dim=checkpoint["edge_dim"],
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            heads=config.heads,
            dropout=config.dropout,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return model, config

    def load_resume_state(
        self,
        device: torch.device,
    ) -> dict[str, object] | None:
        if not self.latest_checkpoint_path.exists():
            return None

        return self.load_checkpoint(device=device, checkpoint_path=self.latest_checkpoint_path)

    def save_test_metrics(self, metrics: dict[str, float]) -> None:
        self.test_metrics_path.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
