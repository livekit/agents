"""The provider supplies wire rules; the harness supplies scenarios and faults."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any

from aiohttp import WSMessage, web

if TYPE_CHECKING:
    from livekit.agents import stt

    from .faults import TransportFixture
    from .server import Connection, STTServer


@dataclass(frozen=True)
class Transcript:
    text: str
    final: bool
    start: float
    end: float
    index: int


class Protocol:
    """One independent server-side session. Never import a plugin's wire helpers."""

    def __init__(self, connection: Connection):
        self.connection = connection
        self.configured = False

    async def open(self, request: web.Request) -> None:
        pass

    async def receive(self, message: WSMessage) -> None:
        raise NotImplementedError

    async def send_transcript(self, result: Transcript) -> None:
        raise NotImplementedError

    async def finish_turn(self) -> None:
        await self.connection.finalize()


@dataclass(frozen=True, kw_only=True)
class ProviderConfig:
    """Shared discovery metadata; production dependencies are imported lazily."""

    live_constructor: str = ""
    required_env: tuple[str, ...] = ()
    live_options: dict[str, Any] = field(default_factory=dict)
    live_options_from_env: dict[str, str] = field(default_factory=dict)
    transport_fixture: TransportFixture | None = None

    def create_live_stt(self):
        module, constructor = self.live_constructor.split(":")
        options = {
            **self.live_options,
            **{arg: os.environ[name] for arg, name in self.live_options_from_env.items()},
        }
        return getattr(import_module(module), constructor)(**options)


@dataclass(frozen=True)
class Provider(ProviderConfig):
    name: str
    protocol: type[Protocol]
    create_stt: Callable[[STTServer, Any], stt.STT]
    sources: tuple[str, ...]
    sample_rate: int = 16000
    # Only a missing protocol guarantee can exclude a completion test.
    completion_unknown_reason: str = ""
    # Some protocols have turn finals but no acknowledgement for an idle stop.
    require_completion: bool = True
    completion_on_close: bool = True
    # SDKs may manage heartbeat themselves rather than use APIConnectOptions.
    uses_websocket_monitor: bool = True
    final_response_timeout: float = 0.25
    bootstrap: Callable[..., Any] | None = None
    tls: bool = False
    expected_attempts_after_audio: int = 3
    # A replay buffer can require a final before its separate replay debt clears.
    progress_requires_final: bool = False
    reconnect_on_idle_close: bool = False


@dataclass(frozen=True)
class SDKProvider(ProviderConfig):
    name: str
    create_fixture: Callable[..., Any]
    sources: tuple[str, ...]
    minimum_python: tuple[int, int] = (3, 10)
