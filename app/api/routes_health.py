"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness/readiness probe.

    Returns ``{"status":"ok"}`` when the process is up and ready. Includes
    engine/model metadata and model download phase when available.
    """
    engine = getattr(request.app.state, "engine", None)
    settings = getattr(request.app.state, "settings", None)
    model_state = getattr(request.app.state, "model_state", None)

    ready = False
    engine_name = None
    model_name = None
    model_phase = None
    model_ready = None

    if model_state is not None:
        snap = model_state.snapshot()
        model_phase = snap.get("phase")
        model_ready = bool(snap.get("ready"))
        model_name = snap.get("model")
        ready = model_ready

    if engine is not None:
        engine_ready = bool(engine.is_ready())
        ready = ready and engine_ready if model_state is not None else engine_ready
        engine_name = engine.name
        model_name = getattr(engine, "model_id", None) or model_name

    if settings is not None and model_name is None:
        model_name = settings.resolved_model_name

    if model_state is None and engine is None:
        # Legacy / minimal apps: process up
        ready = True

    return HealthResponse(
        status="ok" if ready else "starting",
        engine=engine_name,
        model=model_name,
        ready=ready,
        model_phase=model_phase,
        model_ready=model_ready,
    )
