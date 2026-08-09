"""TTS HTTP endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from app.models.schemas import TTSRequest
from app.services.tts_service import TTSService, TTSServiceError

router = APIRouter(tags=["tts"])


def _service(request: Request) -> TTSService:
    svc = getattr(request.app.state, "tts_service", None)
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="TTS engine not ready; see GET /model/status",
        )
    return svc  # type: ignore[no-any-return]


def _audio_response(result) -> Response:
    headers = {
        "X-Chunk-Count": str(result.chunk_count),
        "X-Generation-Time-Ms": str(result.generation_time_ms),
        "X-Model": result.model,
        "X-Engine": result.engine,
        "X-Sample-Rate": str(result.sample_rate),
        "X-Language": result.language,
        "X-Speaker": result.speaker,
        "Content-Disposition": f'attachment; filename="{result.filename}"',
    }
    return Response(
        content=result.content,
        media_type=result.media_type,
        headers=headers,
    )


def _raise_service_error(exc: TTSServiceError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post("/tts")
async def tts_json(body: TTSRequest, request: Request) -> Response:
    """Synthesize speech from a JSON body and return audio bytes."""
    svc = _service(request)
    try:
        result = svc.synthesize(body)
    except TTSServiceError as exc:
        _raise_service_error(exc)
    return _audio_response(result)


@router.post("/tts-file")
async def tts_file(
    request: Request,
    file: UploadFile = File(..., description="UTF-8 text file to read aloud"),
    language: str = Form(default="yue"),
    speaker: str = Form(default="default"),
    speed: float = Form(default=1.0),
    format: Literal["wav", "mp3"] = Form(default="wav"),
) -> Response:
    """Accept a text file upload, synthesize speech, return audio."""
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="file must be UTF-8 encoded text",
        ) from exc

    svc = _service(request)
    try:
        result = svc.synthesize_file_text(
            text,
            language=language,
            speaker=speaker,
            speed=speed,
            fmt=format,
            source_filename=file.filename,
        )
    except TTSServiceError as exc:
        _raise_service_error(exc)
    return _audio_response(result)


@router.post("/tts-long")
async def tts_long(request: Request) -> Response:
    """Long-form / article TTS.

    Automatically chunks the input, synthesizes each segment, merges audio,
    and returns a single MP3.

    Accepts either:
    - ``application/json`` body: ``{"text","language","speaker","speed"}``
    - ``multipart/form-data`` with ``text`` and/or uploaded ``file``
    """
    content_type = (request.headers.get("content-type") or "").lower()
    language = "yue"
    speaker = "default"
    speed = 1.0
    content = ""

    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
        content = str(payload.get("text") or "").strip()
        language = str(payload.get("language") or language)
        speaker = str(payload.get("speaker") or speaker)
        try:
            speed = float(payload.get("speed") if payload.get("speed") is not None else speed)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="speed must be a number") from exc
    else:
        form = await request.form()
        raw_text = form.get("text")
        if raw_text is not None:
            content = str(raw_text).strip()
        language = str(form.get("language") or language)
        speaker = str(form.get("speaker") or speaker)
        try:
            speed = float(form.get("speed") or speed)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="speed must be a number") from exc

        upload = form.get("file")
        if upload is not None and hasattr(upload, "read"):
            raw = await upload.read()  # type: ignore[misc]
            if raw:
                try:
                    content = raw.decode("utf-8").strip()
                except UnicodeDecodeError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail="file must be UTF-8 encoded text",
                    ) from exc

    if not content:
        raise HTTPException(
            status_code=400,
            detail="provide text via form field, JSON body, or uploaded file",
        )

    svc = _service(request)
    try:
        result = svc.synthesize_long(
            content,
            language=language,
            speaker=speaker,
            speed=speed,
            fmt="mp3",
        )
    except TTSServiceError as exc:
        _raise_service_error(exc)
    return _audio_response(result)
