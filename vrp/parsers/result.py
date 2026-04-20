from __future__ import annotations

from pyvrp import Result

from vrp.models.instance import Instance
from vrp.models.solution import SolutionResult


def parse_result_to_solution_result(
    instance: Instance, result: Result
) -> SolutionResult:
    edge_scores = [0.0] * (instance.get_node_count() ** 2)

    for route in result.best.routes():
        previous_idx = route.start_depot()

        for visit in route.visits():
            current_idx = int(visit)
            edge_scores[get_edge_score_index(instance, previous_idx, current_idx)] = 1.0
            previous_idx = current_idx

        edge_scores[get_edge_score_index(instance, previous_idx, route.end_depot())] = (
            1.0
        )

    return SolutionResult(edge_scores=edge_scores)


def get_edge_score_index(instance: Instance, from_idx: int, to_idx: int) -> int:
    return from_idx * instance.get_node_count() + to_idx
