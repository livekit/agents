from __future__ import annotations

import re
from collections.abc import Mapping
from typing import cast

from pydantic import JsonValue

_EXTERNAL_TRACKING_ID_RE = re.compile(r"[\x00-\x1f\x7f,]")
_MODEL_IDENTIFIER_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)+"
    r"(?::[A-Za-z0-9][A-Za-z0-9._-]*)?$"
)

BRIDGE_ERROR_CODE_STATUS: dict[str, int] = {
    "auth_error": 401,
    "config_error": 400,
    "invalid_request": 400,
    "payload_too_large": 413,
    "rate_limit": 429,
    "rate_limit_exceeded": 429,
    "idle_timeout_exceeded": 408,
    "max_connection_duration_exceeded": 408,
    "not_ready": 503,
    "backpressure": 503,
    "stt_metering_unavailable": 503,
    "translation_error": 500,
    "provider_error": 502,
    "backend_error": 502,
    "backend_connection_failed": 502,
}


def bridge_error_status(code: object) -> int | None:
    if isinstance(code, str):
        return BRIDGE_ERROR_CODE_STATUS.get(code.strip().lower())
    return None


def extract_error_status(frame: Mapping[str, object]) -> int | None:
    candidates: list[object] = []
    nested = frame.get("data")
    if isinstance(nested, Mapping):
        candidates.extend(
            cast(Mapping[object, object], nested).get(key)
            for key in ("status_code", "code", "err_code")
        )
    candidates.extend(frame.get(key) for key in ("status_code", "code", "err_code"))
    for value in candidates:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                status = bridge_error_status(value)
                if status is not None:
                    return status
    return None


def validate_model_identifier(model: object) -> str:
    if not isinstance(model, str) or not _MODEL_IDENTIFIER_RE.fullmatch(model):
        raise ValueError("model must be a provider/model identifier, for example 'deepgram/nova:3'")
    return model


def build_tts_init_payload(
    *,
    model: str,
    voice: str,
    language: str,
    sample_rate: int,
    encoding: str,
    speed: float,
    model_options: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    config: dict[str, JsonValue] = {
        "language": language,
        "encoding": encoding,
        "sample_rate": sample_rate,
        "speed": speed,
        **dict(model_options or {}),
    }
    return {
        "type": "init",
        "model": model,
        "voice": voice,
        "language": language,
        "config": config,
    }


def build_stt_init_payload(
    *,
    model: str | None,
    language: str,
    sample_rate: int,
    encoding: str,
    vad_threshold: float,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
    enable_diarization: bool,
    enable_partial_transcripts: bool,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    model_options: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    del model
    config: dict[str, JsonValue] = {
        "language": language,
        "sample_rate": sample_rate,
        "encoding": "linear16" if encoding == "pcm_s16le" else encoding,
        "vad_threshold": vad_threshold,
        "vad_min_silence_duration_ms": vad_min_silence_duration_ms,
        "vad_speech_pad_ms": vad_speech_pad_ms,
        "enable_diarization": enable_diarization,
        "enable_partials": enable_partial_transcripts,
        "enable_partial_transcripts": enable_partial_transcripts,
        **dict(model_options or {}),
    }
    if min_speakers is not None:
        config["min_speakers"] = min_speakers
    if max_speakers is not None:
        config["max_speakers"] = max_speakers
    partials = config.get(
        "enable_partials",
        config.get("enable_partial_transcripts", enable_partial_transcripts),
    )
    config["enable_partials"] = partials
    config["enable_partial_transcripts"] = partials
    return {"type": "init", "config": config}


def normalize_region_override(region_override: str | list[str] | None) -> str | None:
    if region_override is None:
        return None
    raw_values = (
        region_override.split(",")
        if isinstance(region_override, str)
        else [str(value) for value in region_override]
    )
    values = [value.strip().lower() for value in raw_values if value.strip()]
    return ", ".join(values) or None


def normalize_world_part_override(world_part_override: str | None) -> str | None:
    if world_part_override is None:
        return None
    value = world_part_override.strip().lower()
    return value or None


def _validate_external_tracking_id(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{name} must not be empty")
    if len(cleaned) > 128:
        raise ValueError(f"{name} must be 128 characters or fewer")
    if _EXTERNAL_TRACKING_ID_RE.search(cleaned):
        raise ValueError(f"{name} must not contain commas or control characters")
    return cleaned


def build_external_tracking_headers(
    *,
    external_agent_id: str | None,
    external_session_id: str | None,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    agent_id = _validate_external_tracking_id(
        external_agent_id,
        name="external_agent_id",
    )
    session_id = _validate_external_tracking_id(
        external_session_id,
        name="external_session_id",
    )
    if agent_id is not None:
        headers["X-SLNG-Agent-Id"] = agent_id
    if session_id is not None:
        headers["X-SLNG-Session-Id"] = session_id
    return headers
