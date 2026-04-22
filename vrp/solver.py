from pyvrp.stop import MaxRuntime

from vrp.models.instance import Instance
from vrp.models.solution import SolutionResult

from vrp.parsers.instance import parse_instance_to_model
from vrp.parsers.result import parse_result_to_solution_result


class PyVRPSolver:
    def __init__(
        self,
        max_runtime: float | None = None,
        seed: int | None = None,
        display: bool = False,
    ) -> None:
        self.max_runtime = max_runtime
        self.seed = seed
        self.display = display

    def solve_instance(
        self,
        instance: Instance,
        max_runtime: float | None = None,
    ) -> SolutionResult:
        resolved_max_runtime = (
            max_runtime if max_runtime is not None else self.max_runtime
        )

        if resolved_max_runtime is None:
            raise ValueError("max_runtime must be provided in PyVRPSolver or solve_instance")

        model = parse_instance_to_model(instance)
        solve_kwargs = {
            "stop": MaxRuntime(resolved_max_runtime),
            "display": self.display,
        }

        if self.seed is not None:
            solve_kwargs["seed"] = self.seed

        result = model.solve(
            **solve_kwargs,
        )
        return parse_result_to_solution_result(instance, result)
