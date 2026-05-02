from __future__ import annotations

from pathlib import Path
import sys

import uvicorn

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))


def run_server() -> None:
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    run_server()
