from __future__ import annotations

from fastapi import FastAPI

from server.controllers.analysis_controller import router as analysis_router


app = FastAPI(title="VRP ML API")
app.include_router(analysis_router)
