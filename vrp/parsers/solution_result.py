from vrp.models.gnn_input import GNNInput
from vrp.models.gnn_output import GNNOutput
from vrp.models.instance import Instance
from vrp.models.solution import SolutionResult


def parse_solution_result_to_gnn_output(
    instance: Instance,
    gnn_input: GNNInput,
    solution_result: SolutionResult,
) -> GNNOutput:
    node_count = instance.get_node_count()
    edge_scores = [
        solution_result.edge_scores[from_idx * node_count + to_idx]
        for from_idx, to_idx in gnn_input.get_edge_pairs()
    ]
    return GNNOutput(edge_scores=edge_scores)
