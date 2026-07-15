from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

from .gateway_adapter import validate_model_identifier


@dataclass(frozen=True)
class STTConnectionConfig:
    endpoint: str
    model: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    init: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class TTSConnectionConfig:
    endpoint: str
    model: str | None = None
    voice: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    init: Mapping[str, Any] | None = None
    control_profile: str | None = None


@dataclass(frozen=True)
class PluginEvent:
    name: str
    component: Literal["stt", "tts"]
    level: Literal["info", "warning", "error"] = "info"
    data: Mapping[str, Any] = field(default_factory=dict)


class CandidateState:
    def __init__(self, count: int, recovery_cooldown_s: float) -> None:
        self._count = count
        self._cooldown = max(0.0, recovery_cooldown_s)
        self._active = 0
        self._primary_failed_at: float | None = None

    def start(self) -> int:
        if (
            self._active != 0
            and self._primary_failed_at is not None
            and time.monotonic() - self._primary_failed_at >= self._cooldown
        ):
            self._active = 0
            self._primary_failed_at = None
        return self._active

    def advance(self, index: int) -> int | None:
        if index == 0:
            self._primary_failed_at = time.monotonic()
        next_index = index + 1
        if next_index >= self._count:
            return None
        self._active = next_index
        return next_index

    def select(self, index: int) -> None:
        self._active = index


def bridge_endpoint(base_url: str, service: Literal["stt", "tts"], model: str) -> str:
    validate_model_identifier(model)
    host = base_url.removeprefix("https://").removeprefix("http://").rstrip("/")
    protocol = "ws" if host.split(":", 1)[0] in {"localhost", "127.0.0.1"} else "wss"
    return f"{protocol}://{host}/v1/bridges/unmute/{service}/{model}"


def bridge_model(endpoint: str, service: Literal["stt", "tts"]) -> str:
    parsed = urlparse(endpoint)
    marker = f"/v1/bridges/unmute/{service}/"
    if (
        parsed.scheme not in {"ws", "wss"}
        or not parsed.netloc
        or not parsed.path.startswith(marker)
    ):
        raise ValueError(f"{service.upper()} endpoint must target the Unmute Bridge path {marker}")
    model = parsed.path.split(marker, 1)[1].rstrip("/")
    return validate_model_identifier(model)
