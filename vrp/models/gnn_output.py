class GNNOutput:
    # edge_scores: solution score for each edge
    def __init__(self, edge_scores: list[float]) -> None:
        self.edge_scores = edge_scores

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_scores": self.edge_scores,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "GNNOutput":
        return cls(edge_scores=data["edge_scores"])
