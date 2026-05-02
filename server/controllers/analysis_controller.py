from __future__ import annotations

from fastapi import APIRouter, HTTPException

from server.schemas.gnn import AnalyzeRequest, GnnOutputDto
from vrp.ml.use_model import predict_instance_edges
from vrp.models.gnn_input import GNNInput


router = APIRouter()


@router.post("/analyze", response_model=GnnOutputDto)
def analyze(request: AnalyzeRequest) -> GnnOutputDto:
    try:
        gnn_input = GNNInput.from_dict(request.gnnInput.to_domain_dict())
        gnn_output = predict_instance_edges(gnn_input)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return GnnOutputDto.from_domain_dict(gnn_output.to_dict())
