"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes_health import router as health_router
from app.api.routes_model import router as model_router
from app.api.routes_tts import router as tts_router
from app.config.settings import Settings, get_settings
from app.engines.registry import create_engine
from app.services.audio_service import AudioService
from app.services.model_bootstrap import ModelBootstrap
from app.services.model_download_state import ModelDownloadState, ModelPhase
from app.services.model_manager import ModelManager
from app.services.tts_service import TTSService
from app.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    setup_logging(settings.log_level)
    settings.ensure_directories()

    # Cache dirs for HF / torch on the mounted volume
    import os

    os.environ.setdefault("HF_HOME", str(settings.cache_dir / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(settings.cache_dir / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(settings.cache_dir))

    logger.info(
        "Starting TTS server",
        extra={
            "version": __version__,
            "engine": settings.tts_engine,
            "model": settings.resolved_model_name,
            "device": settings.device,
        },
    )

    state = ModelDownloadState(
        model=settings.resolved_model_name,
        path=str(settings.model_path),
        download_source=settings.download_source,
    )
    manager = ModelManager(settings)

    app.state.model_state = state
    app.state.model_manager = manager
    app.state.engine = None
    app.state.tts_service = None

    def load_engine() -> None:
        engine = create_engine(settings)
        engine.load()
        audio = AudioService()
        tts_service = TTSService(settings=settings, engine=engine, audio=audio)
        app.state.engine = engine
        app.state.tts_service = tts_service
        logger.info(
            "TTS engine loaded",
            extra={"engine": engine.name, "model": engine.model_id},
        )

    bootstrap = ModelBootstrap(
        settings=settings,
        manager=manager,
        state=state,
        load_engine=load_engine,
    )
    app.state.model_bootstrap = bootstrap

    if manager.is_model_present():
        try:
            state.set_phase(ModelPhase.LOADING_ENGINE, message="Loading TTS engine")
            load_engine()
            state.set_phase(ModelPhase.READY, message="Model ready")
        except Exception as exc:
            logger.exception("Engine load failed on startup")
            state.set_error(str(exc))
    elif not settings.skip_model_download:
        logger.info("Model missing or incomplete; starting background download")
        bootstrap.ensure_async()
    else:
        state.set_error(
            f"Model missing at {settings.model_path} and SKIP_MODEL_DOWNLOAD=true"
        )

    try:
        yield
    finally:
        logger.info("Shutting down TTS server")
        engine = getattr(app.state, "engine", None)
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:
                logger.exception("Error during engine shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title="TTS Server",
        description=(
            "Production multi-engine text-to-speech API. "
            "Default engine: CosyVoice 3. Backends are swappable via TTS_ENGINE."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(model_router)
    app.include_router(tts_router)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning(
            "Validation error",
            extra={"path": str(request.url.path), "errors": exc.errors()},
        )
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled error",
            extra={"path": str(request.url.path), "error": str(exc)},
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error"},
        )

    return app


app = create_app()
