# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from typing import Self

from livekit.agents.types import APIConnectOptions

from .._core import ChatItem
from .a2a_auth import AuthGrant, AuthOperation, AuthPrincipal
from .a2a_card import EndpointDescriptor
from .a2a_client import AgentCard, Message, MessageMetadata, TaskUpdate
from .a2a_door import DoorFrame
from .a2a_remote import RemoteEvent

class _core_Credentials:
    def __init__(self, api_key: str, api_secret: str) -> None: ...
    def verify(
        self,
        token: str,
        *,
        agent_name: str = "",
        deployment: str = "",
    ) -> AuthPrincipal: ...
    def principal(
        self,
        authorization: str | None,
        thread_grant: str | None,
        node_authorized: bool,
        agent_name: str,
        deployment: str,
    ) -> AuthPrincipal: ...
    def project_token(self, *, identity: str = "worker", ttl_seconds: int = 3600) -> str: ...
    def thread_grant_token(
        self,
        grant: AuthGrant,
        *,
        identity: str = "",
        ttl_seconds: int = 3600,
    ) -> str: ...

def _core_auth_grant_allows(
    grant: AuthGrant,
    operation: AuthOperation,
    thread_id: str,
    scope: str | None,
) -> bool: ...

def _core_a2a_extensions_header(requested: str | None) -> str | None:
    """Echo the supported capabilities selected by the optional extension header."""

def _core_a2a_agent_card(descriptor: EndpointDescriptor) -> str:
    """Serialize the upstream A2A card; the host only adapts it to its HTTP/SDK representation."""

class Reply:
    @property
    def task_id(self) -> str: ...
    @property
    def thread_id(self) -> str: ...
    def __aiter__(self) -> Self: ...
    async def __anext__(self) -> TaskUpdate: ...
    async def aclose(self) -> None: ...

class AgentClient:
    def __init__(
        self,
        endpoint_url: str,
        *,
        token: str | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> None: ...
    async def connect(self) -> AgentCard: ...
    @property
    def offers_extension(self) -> bool: ...
    @property
    def extensions(self) -> list[str]: ...
    async def send(
        self,
        message: str | Message,
        *,
        thread_id: str | None = None,
        task_id: str | None = None,
        metadata: MessageMetadata | None = None,
        chat_ctx: list[ChatItem] | None = None,
    ) -> Reply: ...
    async def _core_subscribe(self, task_id: str, thread_id: str) -> Reply: ...
    async def cancel(
        self,
        task_id: str,
        *,
        thread_id: str | None = None,
        reason: str | None = None,
    ) -> TaskUpdate: ...
    async def aclose(self) -> None: ...

class _core_A2ADoor:
    @property
    def max_payload_bytes(self) -> int: ...
    async def dispatch(
        self,
        body: bytes | bytearray | list[int],
        principal: AuthPrincipal,
        *,
        version: str | None = None,
        offset: str | None = None,
        body_size: int | None = None,
        host_context: object | None = None,
    ) -> _core_A2AResponse: ...
    async def aclose(self) -> None: ...

class _core_A2AResponse:
    @property
    def streaming(self) -> bool: ...
    async def next(self) -> DoorFrame | None: ...
    async def aclose(self) -> None: ...

class _core_RemoteInteraction:
    def __init__(
        self,
        endpoint: str,
        tool_id: str,
        function_call: ChatItem,
        history: list[ChatItem],
    ) -> None: ...
    def message(
        self,
        arguments: object,
        structured: bool,
        offers_extension: bool,
        history: list[ChatItem],
    ) -> Message: ...
    def update(self, update: TaskUpdate) -> RemoteEvent: ...
    def cancelled(self) -> None: ...
    @property
    def extra(self) -> dict[str, object]: ...

def _core_remote_token(endpoint: str, token: str | None = None) -> str | None:
    """Resolve an endpoint credential from explicit input or the configured project's origin."""
