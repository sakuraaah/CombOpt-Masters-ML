from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from vrp.dataset.storage import TrainingDatasetStorage
from vrp.utils.visualizer import plot_graph_pair


DATASET_NAME = "vrp_training_dataset_v2"
SHARD_SIZE = 100
SAMPLE_ID = 150


if __name__ == "__main__":
    dataset_dir = (
        Path(__file__).resolve().parents[1] / "dataset" / "data" / DATASET_NAME
    )
    storage = TrainingDatasetStorage(dataset_dir=dataset_dir, shard_size=SHARD_SIZE)
    sample = storage.load_sample(SAMPLE_ID)
    plot_graph_pair(sample.gnn_input, sample.gnn_output)
