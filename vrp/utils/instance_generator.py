import math
import random

from vrp.models.instance import DistanceMatrix, Instance, Node


def distance(a: Node, b: Node) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def generate_instance(n: int, seed: int | None = None) -> Instance:
    rng = random.Random(seed)
    nodes: list[Node] = []

    num_couriers = rng.randint(1, max(1, n // 10))
    num_clients = n - num_couriers

    for idx in range(num_couriers):
        nodes.append(
            {
                "id": idx,
                "type": "depot",
                "courier_id": idx,
                "x": rng.uniform(0, 100),
                "y": rng.uniform(0, 100),
            }
        )

    for idx in range(num_clients):
        nodes.append(
            {
                "id": num_couriers + idx,
                "type": "client",
                "courier_id": None,
                "x": rng.uniform(0, 100),
                "y": rng.uniform(0, 100),
            }
        )

    distance_matrix: DistanceMatrix = []

    for i in range(n):
        row: list[float] = []

        for j in range(n):
            row.append(distance(nodes[i], nodes[j]))

        distance_matrix.append(row)

    return Instance(
        nodes=nodes,
        distance_matrix=distance_matrix,
    )
