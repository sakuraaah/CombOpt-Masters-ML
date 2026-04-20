from __future__ import annotations

from vrp.models.gnn_output import GNNOutput
from vrp.models.solution import SolutionResult


def parse_solution_result_to_gnn_output(solution_result: SolutionResult) -> GNNOutput:
    return GNNOutput(edge_scores=solution_result.edge_scores)
