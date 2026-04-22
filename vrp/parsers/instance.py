from pyvrp import Model

from vrp.constants import GRAPH_K
from vrp.models.gnn_input import GNNInput
from vrp.models.instance import Instance


def parse_instance_to_model(instance: Instance) -> Model:
    model = Model()
    depots = []

    for node in instance.nodes:
        if node["type"] == "depot":
            depots.append(model.add_depot(x=node["x"], y=node["y"]))
        elif node["type"] == "client":
            model.add_client(x=node["x"], y=node["y"])
        else:
            raise ValueError(f"Unknown node type: {node['type']}")

    for depot in depots:
        model.add_vehicle_type(
            num_available=1,
            start_depot=depot,
            end_depot=depot,
        )

    pyvrp_locations = list(model.locations)

    for from_idx, from_location in enumerate(pyvrp_locations):
        for to_idx, to_location in enumerate(pyvrp_locations):
            distance = round(instance.distance_matrix[from_idx][to_idx])
            model.add_edge(from_location, to_location, distance=distance)

    return model


def parse_instance_to_gnn_input(instance: Instance) -> GNNInput:
    max_coordinate = get_max_coordinate(instance)
    coordinate_scale = max_coordinate if max_coordinate > 0 else 1.0

    node_features: list[list[float]] = []

    for node in instance.nodes:
        node_features.append(
            [
                node["x"] / coordinate_scale,
                node["y"] / coordinate_scale,
                1.0 if node["type"] == "depot" else 0.0,
                float(node["courier_id"]) if node["courier_id"] is not None else -1.0,
            ]
        )

    edge_pairs = get_trimmed_edge_pairs(instance)
    edge_index: list[list[int]] = [
        [from_idx for from_idx, _ in edge_pairs],
        [to_idx for _, to_idx in edge_pairs],
    ]
    edge_features = [
        [instance.distance_matrix[from_idx][to_idx]]
        for from_idx, to_idx in edge_pairs
    ]

    return GNNInput(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
    )


def get_max_coordinate(instance: Instance) -> float:
    return max(max(node["x"], node["y"]) for node in instance.nodes)


def get_trimmed_edge_pairs(instance: Instance) -> list[tuple[int, int]]:
    node_count = instance.get_node_count()

    if node_count <= 1:
        return []

    k_neighbors = min(GRAPH_K, node_count - 1)
    depot_indices = {
        node_idx
        for node_idx, node in enumerate(instance.nodes)
        if node["type"] == "depot"
    }
    edge_pairs: set[tuple[int, int]] = set()

    for from_idx in range(node_count):
        if from_idx in depot_indices:
            for to_idx in range(node_count):
                if to_idx != from_idx:
                    edge_pairs.add((from_idx, to_idx))
                    edge_pairs.add((to_idx, from_idx))
            continue

        nearest_targets = sorted(
            (
                (instance.distance_matrix[from_idx][to_idx], to_idx)
                for to_idx in range(node_count)
                if to_idx != from_idx
            ),
            key=lambda item: item[0],
        )

        for _, to_idx in nearest_targets[:k_neighbors]:
            edge_pairs.add((from_idx, to_idx))

        for depot_idx in depot_indices:
            edge_pairs.add((from_idx, depot_idx))
            edge_pairs.add((depot_idx, from_idx))

    return sorted(edge_pairs)
