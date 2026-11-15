from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

from pydantic import JsonValue

from .gateway_adapter import validate_model_identifier


@dataclass(frozen=True)
class TTSConnectionConfig:
    endpoint: str
    model: str | None = None
    voice: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict[str, str])
    init: Mapping[str, JsonValue] | None = None


@dataclass(frozen=True)
class PluginEvent:
    name: str
    component: Literal["stt", "tts"]
    level: Literal["info", "warning", "error"] = "info"
    data: Mapping[str, object] = field(default_factory=dict[str, object])


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
