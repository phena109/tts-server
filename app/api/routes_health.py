"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness/readiness probe.

    Returns ``{"status":"ok"}`` when the process is up. Includes engine/model
    metadata when the app has finished startup loading.
    """
    engine = getattr(request.app.state, "engine", None)
    settings = getattr(request.app.state, "settings", None)
    ready = True
    engine_name = None
    model_name = None

    if engine is not None:
        ready = bool(engine.is_ready())
        engine_name = engine.name
        model_name = getattr(engine, "model_id", None)
    if settings is not None and model_name is None:
        model_name = settings.resolved_model_name

    return HealthResponse(
        status="ok" if ready else "starting",
        engine=engine_name,
        model=model_name,
        ready=ready,
    )
