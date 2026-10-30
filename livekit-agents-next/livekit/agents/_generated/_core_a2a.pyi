# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

from livekit.agents.storage.context import _TurnHost
from livekit.agents.types import APIConnectOptions

from .._core import Storage
from .a2a_auth import AuthGrant, AuthOperation, AuthPrincipal
from .a2a_card import EndpointDescriptor
from .a2a_client import AgentCard, Message, MessageInput, MessageMetadata, TaskUpdate
from .a2a_door import DoorConfig, DoorFrame
from .a2a_remote import RemoteEvent
from .chat_items import ChatItem

class _core_Credentials:
    def __new__(cls, api_key: str, api_secret: str) -> Self: ...
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

def _core_a2a_message_text(message: Message) -> str:
    """The text parts of a generated A2A message, joined by newlines."""

def _core_a2a_message_data(message: Message) -> dict[str, object] | None:
    """The first object data part of a generated A2A message."""

class Reply:
    @property
    def task_id(self) -> str: ...
    @property
    def thread_id(self) -> str: ...
    def __aiter__(self) -> Self: ...
    async def __anext__(self) -> TaskUpdate: ...
    async def aclose(self) -> None: ...

class AgentClient:
    def __new__(
        cls,
        endpoint_url: str,
        *,
        token: str | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> Self: ...
    async def connect(self) -> AgentCard: ...
    @property
    def offers_extension(self) -> bool: ...
    async def send(
        self,
        message: MessageInput,
        *,
        thread_id: str | None = None,
        task_id: str | None = None,
        metadata: MessageMetadata | None = None,
        chat_ctx: Sequence[ChatItem] | None = None,
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
    def __new__(cls, storage: Storage, config: DoorConfig, host: _TurnHost) -> Self:
        """The door over `storage`'s namespace, which opens its backend on first use; accepted work
        is settled by the storage's `aclose`.
        """
    @property
    def max_payload_bytes(self) -> int: ...
    async def dispatch(
        self,
        body: bytes | bytearray | memoryview | list[int],
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
    def __new__(
        cls,
        endpoint: str,
        tool_id: str,
        function_call: ChatItem,
        history: Sequence[ChatItem],
    ) -> Self: ...
    def message(
        self,
        arguments: object,
        structured: bool,
        offers_extension: bool,
        history: Sequence[ChatItem],
        thread_id: str | None = None,
        token: str | None = None,
    ) -> Message:
        """The call's message; `thread_id` shares the caller's thread under `token`'s grant for it."""
    def answer(self, request_id: str, text: str) -> Message:
        """The person's answer to the retained input request `request_id`, on its task; refused
        once that request was settled or superseded.
        """
    def update(self, update: TaskUpdate) -> RemoteEvent: ...
    def cancelled(self) -> None: ...
    @property
    def extra(self) -> dict[str, object]: ...

def _core_remote_token(
    endpoint: str,
    token: str | None = None,
    thread_id: str | None = None,
) -> str | None:
    """Resolve an endpoint credential: `token` as given; else, for an endpoint on the configured
    project's origin, a grant for the caller's `thread_id`, or a project credential without one.
    """
