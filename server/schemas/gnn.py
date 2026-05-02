from __future__ import annotations

from pydantic import BaseModel


class GnnInputDto(BaseModel):
    node_features: list[list[float]]
    edge_index: list[list[int]]
    edge_features: list[list[float]]

    def to_domain_dict(self) -> dict[str, object]:
        return _model_to_dict(self)


class AnalyzeRequest(BaseModel):
    gnnInput: GnnInputDto


class GnnOutputDto(BaseModel):
    edge_scores: list[float]

    @classmethod
    def from_domain_dict(cls, data: dict[str, object]) -> "GnnOutputDto":
        return cls(**data)


def _model_to_dict(model: BaseModel) -> dict[str, object]:
    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()
