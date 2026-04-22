from vrp.models.gnn_input import GNNInput
from vrp.models.gnn_output import GNNOutput


class Sample:
    def __init__(
        self,
        sample_id: int,
        problem_size: int,
        max_runtime: float,
        gnn_input: GNNInput,
        gnn_output: GNNOutput,
    ) -> None:
        self.sample_id = sample_id
        self.problem_size = problem_size
        self.max_runtime = max_runtime
        self.gnn_input = gnn_input
        self.gnn_output = gnn_output

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "problem_size": self.problem_size,
            "max_runtime": self.max_runtime,
            "input": self.gnn_input.to_dict(),
            "output": self.gnn_output.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Sample":
        return cls(
            sample_id=data["sample_id"],
            problem_size=data["problem_size"],
            max_runtime=data["max_runtime"],
            gnn_input=GNNInput.from_dict(data["input"]),
            gnn_output=GNNOutput.from_dict(data["output"]),
        )
