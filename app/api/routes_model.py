"""Model download status and ensure endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.models.schemas import ModelStatusResponse

router = APIRouter(tags=["model"])


def _state(request: Request):
    return request.app.state.model_state


def _bootstrap(request: Request):
    return getattr(request.app.state, "model_bootstrap", None)


@router.get("/model/status", response_model=ModelStatusResponse)
async def model_status(request: Request) -> ModelStatusResponse:
    """Return a snapshot of model download / readiness state."""
    snap = _state(request).snapshot()
    return ModelStatusResponse(**snap)


@router.post("/model/ensure", response_model=ModelStatusResponse)
async def model_ensure(request: Request) -> JSONResponse:
    """Idempotently start or resume model ensure (download + engine load)."""
    boot = _bootstrap(request)
    if boot is None:
        raise HTTPException(status_code=503, detail="model bootstrap not configured")
    result = boot.ensure_async()
    snap = _state(request).snapshot()
    if result.get("conflict"):
        body = {**snap, "already_running": False, "already_ready": False}
        return JSONResponse(status_code=409, content=body)
    status_code = 202 if result.get("started") else 200
    body = {
        **snap,
        "already_running": bool(result.get("already_running")),
        "already_ready": bool(result.get("already_ready")),
    }
    return JSONResponse(status_code=status_code, content=body)
