from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from vrp.utils.instance_generator import generate_instance
from vrp.parsers.instance import parse_instance_to_gnn_input
from vrp.parsers.solution_result import parse_solution_result_to_gnn_output
from vrp.solver import PyVRPSolver
from vrp.utils.visualizer import plot_graph_pair


SEED = 253
PROBLEM_SIZE = 20
MAX_RUNTIME = 0.1


if __name__ == "__main__":
    instance = generate_instance(PROBLEM_SIZE, seed=SEED)
    solver = PyVRPSolver(max_runtime=MAX_RUNTIME, seed=SEED, display=True)

    solution_result = solver.solve_instance(instance)

    graph_input = parse_instance_to_gnn_input(instance)
    graph_output = parse_solution_result_to_gnn_output(solution_result)

    plot_graph_pair(
        graph_input, graph_output, file_name_prefix=f"opt_n{PROBLEM_SIZE}_s{SEED}"
    )
