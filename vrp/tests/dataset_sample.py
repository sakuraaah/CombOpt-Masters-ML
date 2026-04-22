from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from vrp.constants import DATASET_DIR
from vrp.dataset.storage import TrainingDatasetStorage
from vrp.utils.visualizer import plot_graph_pair


SAMPLE_ID = 3646


if __name__ == "__main__":
    storage = TrainingDatasetStorage(dataset_dir=DATASET_DIR)
    sample = storage.load_sample(SAMPLE_ID)
    plot_graph_pair(
        sample.gnn_input,
        sample.gnn_output,
        file_name_prefix=f"ds_id{SAMPLE_ID}_n{sample.problem_size}",
    )
