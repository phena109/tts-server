"""FastAPI dependency helpers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Request

from app.config.settings import Settings
from app.services.tts_service import TTSService


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def get_tts_service(request: Request) -> TTSService:
    return request.app.state.tts_service  # type: ignore[no-any-return]


SettingsDep = Annotated[Settings, ...]
TTSServiceDep = Annotated[TTSService, ...]
