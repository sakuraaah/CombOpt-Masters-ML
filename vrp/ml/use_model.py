from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch

from vrp.constants import OUTPUT_DIR
from vrp.ml.storage import MLTrainingStorage
from vrp.models.gnn_input import GNNInput
from vrp.models.gnn_output import GNNOutput
from vrp.parsers.gnn_input import parse_gnn_input_to_data


def predict_instance_edges(
    gnn_input: GNNInput,
) -> GNNOutput:
    storage = MLTrainingStorage(OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = storage.load_model(device=device)

    data = parse_gnn_input_to_data(gnn_input)
    data = data.to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(data)).cpu()

    return GNNOutput(edge_scores=probs.tolist())
