from pathlib import Path

GRAPH_K = 20
SHARD_SIZE = 100
DATASET_NAME = "vrp_training_dataset_v4"
OUTPUT_NAME = "edge_gnn"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / DATASET_NAME
OUTPUT_DIR = PROJECT_ROOT / "runs" / OUTPUT_NAME
CHECKPOINT_PATH = OUTPUT_DIR / "best.pt"
PLOT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "plots"
