from __future__ import annotations

from pyvrp import Model

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
    edge_index: list[list[int]] = [[], []]
    edge_features: list[list[float]] = []

    for node in instance.nodes:
        node_features.append(
            [
                node["x"] / coordinate_scale,
                node["y"] / coordinate_scale,
                1.0 if node["type"] == "depot" else 0.0,
                float(node["courier_id"]) if node["courier_id"] is not None else -1.0,
            ]
        )

    for from_idx in range(instance.get_node_count()):
        for to_idx in range(instance.get_node_count()):
            edge_index[0].append(from_idx)
            edge_index[1].append(to_idx)
            edge_features.append([instance.distance_matrix[from_idx][to_idx]])

    return GNNInput(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
    )


def get_max_coordinate(instance: Instance) -> float:
    return max(max(node["x"], node["y"]) for node in instance.nodes)
