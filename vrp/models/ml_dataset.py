from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from vrp.dataset.storage import TrainingDatasetStorage
from vrp.models.sample import Sample
from vrp.parsers.gnn_input import parse_gnn_input_to_data


def get_split_name(sample_id: int) -> str:
    remainder = sample_id % 10
    if remainder < 8:
        return "train"
    if remainder == 8:
        return "val"
    return "test"


class MLDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str) -> None:
        super().__init__()
        self.storage = TrainingDatasetStorage(dataset_dir=dataset_dir)

        entries = self.storage.build_or_load_index()
        self.entries = [
            entry for entry in entries if get_split_name(entry.sample_id) == split
        ]

        if not self.entries:
            raise ValueError(f"No samples found for split={split!r} in {dataset_dir}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Data:
        record = self.storage.read_indexed_record(self.entries[idx])
        sample = Sample.from_dict(record)

        y = torch.tensor(sample.gnn_output.edge_scores, dtype=torch.float32)
        data = parse_gnn_input_to_data(sample.gnn_input)
        data.y = y
        data.sample_id = torch.tensor([sample.sample_id], dtype=torch.long)
        data.problem_size = torch.tensor([sample.problem_size], dtype=torch.long)
        return data
