import random

from vrp.dataset.utils import get_max_runtime
from vrp.models.sample import Sample
from vrp.utils.instance_generator import generate_instance
from vrp.parsers.instance import parse_instance_to_gnn_input
from vrp.parsers.solution_result import parse_solution_result_to_gnn_output
from vrp.solver import PyVRPSolver


def build_training_sample(sample_id: int, solver: PyVRPSolver) -> Sample:
    problem_size = random.randint(10, 100)
    max_runtime = get_max_runtime(problem_size)

    instance = generate_instance(problem_size)
    solution_result = solver.solve_instance(instance, max_runtime=max_runtime)

    gnn_input = parse_instance_to_gnn_input(instance)
    gnn_output = parse_solution_result_to_gnn_output(
        instance,
        gnn_input,
        solution_result,
    )

    return Sample(
        sample_id=sample_id,
        problem_size=problem_size,
        max_runtime=max_runtime,
        gnn_input=gnn_input,
        gnn_output=gnn_output,
    )
