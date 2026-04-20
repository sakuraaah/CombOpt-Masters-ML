from __future__ import annotations

from typing import Literal, TypedDict


NodeType = Literal["depot", "client"]


class Node(TypedDict):
    id: int
    type: NodeType
    courier_id: int | None
    x: float
    y: float


DistanceMatrix = list[list[float]]


class Instance:
    def __init__(self, nodes: list[Node], distance_matrix: DistanceMatrix) -> None:
        self.nodes = nodes
        self.distance_matrix = distance_matrix

    def get_node_count(self) -> int:
        return len(self.nodes)
