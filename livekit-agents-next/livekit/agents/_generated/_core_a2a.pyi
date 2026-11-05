# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

from livekit.agents.a2a import _Delegation
from livekit.agents.storage.session import _TurnHost

from .._core import Storage, Thread
from .a2a_auth import AgentThreadGrant, AuthOperation, Principal
from .a2a_card import EndpointDescriptor
from .a2a_client import AgentCard, Message, MessageInput, MessageMetadata, TaskUpdate
from .a2a_door import DoorConfig
from .a2a_remote import RemoteEvent
from .chat_items import ChatItem
from .connect import APIConnectOptions
from .storage_handler import HandlerOutcome, HandlerRun
from .storage_options import TurnOptions
from .storage_turn import HandlerOutput, TurnResult

class _core_Credentials:
    def __new__(cls, api_key: str, api_secret: str) -> Self: ...
    def verify(self, token: str, *, agent_name: str = "", deployment: str = "") -> Principal: ...
    def principal(
        self,
        authorization: str | None,
        thread_grant: str | None,
        node_authorized: bool,
        agent_name: str,
        deployment: str,
    ) -> Principal: ...
    def project_token(self, *, identity: str = "worker", ttl_seconds: int = 3600) -> str: ...
    def thread_grant_token(
        self,
        grant: AgentThreadGrant,
        *,
        identity: str = "",
        ttl_seconds: int = 3600,
    ) -> str: ...

def _core_auth_grant_allows(
    grant: AgentThreadGrant,
    operation: AuthOperation,
    thread_id: str,
    scope: str | None,
) -> bool: ...

def _core_unauthenticated_principal() -> Principal:
    """The caller of the console's unauthenticated listener: every thread permission, threads
    created in the `console` scope.
    """

def _core_a2a_agent_card(descriptor: EndpointDescriptor) -> str:
    """The upstream A2A card's JSON, as the endpoint serves it."""

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
    async def aclose(self) -> None: ...

class _core_RemoteInteraction:
    """One delegation of a tool call to a remote agent (docs/agents.md "Remote agents")."""
    def __new__(
        cls,
        endpoint: str,
        tool_id: str,
        function_call: ChatItem,
        history: Sequence[ChatItem],
        project_url: str,
        api_key: str,
        api_secret: str,
    ) -> Self:
        """`project_url`, `api_key` and `api_secret` are the server's LiveKit project, which grants
        and verifies a shared thread; empty when there is none.
        """
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
    async def run(
        self,
        host: _Delegation,
        arguments: object,
        structured: bool,
        history: Sequence[ChatItem],
        token: str | None,
        thread_id: str | None,
        conn_options: APIConnectOptions | None,
        live: bool,
    ) -> RemoteEvent:
        """Run the delegation on the endpoint: connect with `token`, send the call's message
        (`thread_id` shares the caller's thread under `token`'s grant), then follow its updates:
        each reaches `host.progress`, a moved task is followed, and on a call (`live`) the
        request is announced through `host.delegated` and each question answered through
        `host.ask`. Returns the outcome, or, off a call, the expert's question. The client and
        its reply are closed before it returns.
        """
    def cancelled(self) -> None:
        """The tool call was cancelled: the interaction records it and a running delegation stops,
        closing its reply (which cancels the remote task) and client; `settled()` waits for that.
        """
    async def settled(self) -> None:
        """After `cancelled()`: the running delegation's cleanup, bounded by the teardown limit."""
    @property
    def extra(self) -> dict[str, object]: ...

def _core_delegation_tool(name: str, description: str) -> AgentCard:
    """The delegation tool declared in code: `name` and `description` are required, and the tool
    takes the request, stated in full. Nothing is fetched from the endpoint.
    """

def _core_remote_token(
    endpoint: str,
    project_url: str,
    api_key: str,
    api_secret: str,
    token: str | None = None,
    thread_id: str | None = None,
) -> str | None:
    """Resolve an endpoint credential: `token` as given; else, for an endpoint on the server's
    project's origin (`project_url`, signed with `api_key`/`api_secret`; empty when there is
    none), a grant for the caller's `thread_id`, or a project credential without one.
    """

class _core_HandlerTurn:
    """One turn as its handler sees it: the handler awaits each report, and the turn reads them in
    that order, publishing progress and tokens and answering checkpoints.

    A report waits while the turn holds 64 unread ones, so a handler never runs ahead of the
    turn. `checkpoint()` returns `True` when the run should yield (a newer message waits under
    `interrupt`, or the client cancelled the task). Once the turn ended, or was closed, reports
    are dropped and `checkpoint()` returns `True`.
    """
    async def checkpoint(self) -> bool: ...
    async def progress(self, text: str) -> None: ...
    async def token(self, delta: str) -> None: ...

class Turn:
    def __new__(
        cls,
        thread: Thread,
        lane: str,
        host: _TurnHost,
        options: TurnOptions | None = None,
    ) -> Self: ...
    @property
    def lane(self) -> str: ...
    def policy_for(self, channel: str) -> str: ...
    async def run(self, message: object, task_id: str | None = None) -> TurnResult: ...

def normalize_handler_output(
    output: HandlerOutput,
    outcome: HandlerOutcome,
    runs: Sequence[HandlerRun],
) -> HandlerOutput: ...
