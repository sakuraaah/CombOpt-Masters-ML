from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from vrp.constants import CHECKPOINT_PATH
from vrp.ml.use_model import predict_instance_edges
from vrp.parsers.instance import parse_instance_to_gnn_input
from vrp.utils.instance_generator import generate_instance
from vrp.utils.visualizer import plot_graph_pair


SEED = 253
PROBLEM_SIZE = 20


if __name__ == "__main__":
    instance = generate_instance(PROBLEM_SIZE, seed=SEED)
    gnn_input = parse_instance_to_gnn_input(instance)

    gnn_output = predict_instance_edges(
        gnn_input=gnn_input,
        checkpoint_path=CHECKPOINT_PATH,
    )

    plot_graph_pair(gnn_input, gnn_output)
