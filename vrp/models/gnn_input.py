# NodeFeatures: [
#   normalized x (x / max_coordinate),
#   normalized y (y / max_coordinate),
#   is_depot (0 or 1),
#   courier_id (or -1 for clients),
# ]
NodeFeatures = list[list[float]]

# EdgeIndex: [[from_node_indices], [to_node_indices]]
EdgeIndex = list[list[int]]

# EdgeFeatures: [distance]
EdgeFeatures = list[list[float]]


class GNNInput:
    def __init__(
        self,
        node_features: NodeFeatures,
        edge_index: EdgeIndex,
        edge_features: EdgeFeatures,
    ) -> None:
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features = edge_features

    def get_node_count(self) -> int:
        return len(self.node_features)

    def get_edge_pairs(self) -> list[tuple[int, int]]:
        return list(zip(self.edge_index[0], self.edge_index[1]))

    def get_coordinates(self, node_idx: int) -> tuple[float, float]:
        node_features = self.node_features[node_idx]
        return node_features[0], node_features[1]

    def is_depot(self, node_idx: int) -> bool:
        return bool(self.node_features[node_idx][2])

    def to_dict(self) -> dict[str, object]:
        return {
            "node_features": self.node_features,
            "edge_index": self.edge_index,
            "edge_features": self.edge_features,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "GNNInput":
        return cls(
            node_features=data["node_features"],
            edge_index=data["edge_index"],
            edge_features=data["edge_features"],
        )
