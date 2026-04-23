from dataclasses import dataclass

from vrp.ml.constants import (
    BATCH_SIZE,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    HEADS,
    HIDDEN_DIM,
    LEARNING_RATE,
    LR_PATIENCE,
    NUM_LAYERS,
    NUM_WORKERS,
    POS_WEIGHT_CAP,
    WEIGHT_DECAY,
)


@dataclass
class TrainConfig:
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    hidden_dim: int = HIDDEN_DIM
    num_layers: int = NUM_LAYERS
    heads: int = HEADS
    dropout: float = DROPOUT
    lr: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    epochs: int = EPOCHS
    lr_patience: int = LR_PATIENCE
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    pos_weight_cap: float = POS_WEIGHT_CAP
