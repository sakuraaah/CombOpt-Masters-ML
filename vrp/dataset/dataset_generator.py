from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from vrp.utils.logger import create_logger
from vrp.solver import PyVRPSolver
from vrp.dataset.sample_builder import build_training_sample
from vrp.dataset.storage import TrainingDatasetStorage


TARGET_SAMPLE_COUNT = 200
SHARD_SIZE = 100
DATASET_NAME = "vrp_training_dataset_v2"


def generate_dataset() -> None:
    logger = create_logger("train.generate_training_dataset")
    dataset_dir = Path(__file__).resolve().parent / "data" / DATASET_NAME
    solver = PyVRPSolver(display=False)

    storage = TrainingDatasetStorage(dataset_dir=dataset_dir, shard_size=SHARD_SIZE)
    storage.initialize(
        {
            "dataset_name": DATASET_NAME,
            "target_sample_count": TARGET_SAMPLE_COUNT,
            "problem_size_range": [10, 100],
            "storage_format": "jsonl_shards",
            "shard_size": SHARD_SIZE,
        }
    )

    completed_sample_ids = storage.get_completed_sample_ids()
    storage.write_progress(TARGET_SAMPLE_COUNT, completed_sample_ids)

    logger.info(
        "Starting dataset generation in %s with %d/%d completed samples.",
        dataset_dir,
        len(completed_sample_ids),
        TARGET_SAMPLE_COUNT,
    )

    while len(completed_sample_ids) < TARGET_SAMPLE_COUNT:
        sample_id = storage.get_next_sample_id(completed_sample_ids)
        sample = build_training_sample(sample_id=sample_id, solver=solver)
        storage.append_sample(sample)
        completed_sample_ids.add(sample_id)
        storage.write_progress(TARGET_SAMPLE_COUNT, completed_sample_ids)

        logger.info(
            "Saved sample %d/%d (problem_size=%s, max_runtime=%s).",
            len(completed_sample_ids),
            TARGET_SAMPLE_COUNT,
            sample.problem_size,
            sample.max_runtime,
        )

    logger.info("Dataset generation finished.")


if __name__ == "__main__":
    generate_dataset()
