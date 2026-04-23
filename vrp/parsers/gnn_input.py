from torch_geometric.data import Data

import torch
from vrp.models.gnn_input import GNNInput


def parse_gnn_input_to_data(gnn_input: GNNInput) -> Data:
    x = torch.tensor(gnn_input.node_features, dtype=torch.float32)
    edge_index = torch.tensor(gnn_input.edge_index, dtype=torch.long)
    edge_attr = torch.tensor(gnn_input.edge_features, dtype=torch.float32)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
