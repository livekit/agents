# Copyright 2026 Komaa DigiTech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The messages lane: Microsoft Teams chat, without a bot credential.

Managed connections only. StandIn owns the Teams bot, authenticates the
activity, resolves it to your connection and strips the bot @mention. Your
handler returns text and StandIn performs the Teams send, so your agent never
holds a Bot Framework credential.

Same shape as the call lane: the worker dials OUT and StandIn pushes messages
down that socket, so there is no listener, no port to expose and no tunnel.

    Teams message
         |
         v
    StandIn gateway        (authenticates, normalizes, signs)
         |   pushed down the worker's outbound socket
         v
    ChatChannel            (this class)
         |   your async handler returns reply text
         v
    back up the same socket; StandIn sends it to Teams
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from ._exceptions import StandInError
from ._hmac import SIGNATURE_HEADER, TIMESTAMP_HEADER, now_ms, sign_handshake
from .log import logger

#: chat-schema.yaml SCHEMA_VERSION. A MAJOR version: additive evolution does not
#: bump it, because the schema already requires receivers to ignore unknown
#: fields. An integer above ours therefore means incompatible semantics.
SCHEMA_VERSION = 1

DEFAULT_CHAT_URL = "wss://teams.standin.komaa.com/api/chat/channel"


@dataclass(frozen=True)
class InboundMessage:
    """One user message, already authenticated and resolved to your connection.

    Reserved bot commands are handled by StandIn and never arrive here. In group
    and channel scope only messages that @mention the bot are relayed, and the
    mention is already stripped from ``text``.
    """

    tenant_id: str
    conversation_id: str
    activity_id: str
    scope: str
    text: str
    sender_name: str | None = None
    sender_aad_id: str | None = None
    sender_is_guest: bool = False
    sender_is_linked_owner: bool = False
    attachments: list[dict[str, Any]] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    locale: str | None = None
    #: Submit payload of an Action.Submit on a card this agent sent. ``text`` is
    #: empty on these messages.
    card_action: dict[str, Any] | None = None

    @property
    def is_personal(self) -> bool:
        return self.scope == "personal"


def parse_inbound(body: str) -> InboundMessage:
    """Parse and validate an inbound message. Raises ValueError naming the
    problem; the caller maps that to HTTP 400."""
    try:
        raw = json.loads(body)
    except ValueError as exc:
        raise ValueError("malformed json") from exc
    if not isinstance(raw, dict):
        raise ValueError("body must be an object")
    for key in ("tenantId", "conversationId", "activityId"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise ValueError(f"{key} is required")
    version = raw.get("schemaVersion", SCHEMA_VERSION)
    if isinstance(version, int) and version > SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schemaVersion {version} (this plugin speaks {SCHEMA_VERSION})"
        )
    raw_sender = raw.get("sender")
    sender: dict[str, Any] = raw_sender if isinstance(raw_sender, dict) else {}
    scope = raw.get("scope")
    return InboundMessage(
        tenant_id=raw["tenantId"],
        conversation_id=raw["conversationId"],
        activity_id=raw["activityId"],
        # ChatScope is an OPEN enum: an unknown value relays as a group chat
        # rather than being rejected.
        scope=scope if isinstance(scope, str) and scope else "personal",
        text=raw["text"] if isinstance(raw.get("text"), str) else "",
        sender_name=sender.get("displayName"),
        sender_aad_id=sender.get("aadObjectId"),
        sender_is_guest=bool(sender.get("isGuest", False)),
        sender_is_linked_owner=bool(sender.get("isLinkedOwner", False)),
        attachments=raw["attachments"] if isinstance(raw.get("attachments"), list) else [],
        mentions=raw["mentions"] if isinstance(raw.get("mentions"), list) else [],
        locale=raw.get("locale") if isinstance(raw.get("locale"), str) else None,
        card_action=raw.get("cardAction") if isinstance(raw.get("cardAction"), dict) else None,
    )


def build_reply(message: InboundMessage, text: str, kind: str = "message") -> dict[str, Any]:
    """The gateway-bound reply. tenantId and conversationId echo the inbound
    EXACTLY: the gateway rejects a mismatch, and that check is the cross-tenant
    leak guard the whole relay rests on."""
    reply: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "tenantId": message.tenant_id,
        "conversationId": message.conversation_id,
        "replyToId": message.activity_id,
        "kind": kind,
        "idempotencyKey": f"{message.activity_id}:{kind}",
    }
    if kind != "typing":
        reply["text"] = text
    return reply


class _Seen:
    """At-least-once dedupe on the schema's activityId idempotency key. Bounded
    LRU: an aged-out redelivery running again is acceptable at-least-once
    behaviour, a fresh double-run is not."""

    def __init__(self, capacity: int = 2048) -> None:
        self._capacity = capacity
        self._seen: OrderedDict[str, None] = OrderedDict()

    def mark_first(self, key: str) -> bool:
        if key in self._seen:
            return False
        self._seen[key] = None
        if len(self._seen) > self._capacity:
            self._seen.popitem(last=False)
        return True


class ChatChannel:
    """Answer Microsoft Teams messages with your agent.

    Dialed out from the worker, like the call lane, so nothing listens and there
    is nothing to expose. Managed connections only, and that needs no flag: the
    socket authenticates with your connection secret, so if it opens at all you
    are managed.

    Args:
        respond: async callable taking an :class:`InboundMessage` and returning
            the reply text. An empty string makes the channel say so rather than
            leaving the user watching a typing indicator forever.
        secret: your StandIn connection secret, defaulting to ``STANDIN_SECRET``.
        url: the chat channel URL, defaulting to ``STANDIN_CHAT_URL``.

    Example:
        ```python
        async def on_message(msg: standin.InboundMessage) -> str:
            return f"You said: {msg.text}"


        chat = standin.ChatChannel(respond=on_message)
        await chat.start()
        ```
    """

    #: Serialization means a hung turn would wedge its conversation forever, so
    #: every turn is bounded. Generous: agent turns legitimately run long.
    TURN_TIMEOUT_S = 300.0

    def __init__(
        self,
        *,
        respond: Callable[[InboundMessage], Awaitable[str]],
        secret: str | None = None,
        url: str | None = None,
    ) -> None:
        self._secret = secret or os.environ.get("STANDIN_SECRET", "")
        if not self._secret:
            raise StandInError(
                "a StandIn connection secret is required: pass secret=... or set STANDIN_SECRET"
            )
        self._respond = respond
        self._url = url or os.environ.get("STANDIN_CHAT_URL") or DEFAULT_CHAT_URL
        self._seen = _Seen()
        self._http: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._task: asyncio.Task[Any] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        #: Per-conversation chains. The schema promises per-conversation
        #: ORDERING; independent tasks would let replies overtake each other.
        self._chains: dict[str, asyncio.Task[Any]] = {}
        self._closed = False

    async def start(self) -> None:
        """Dial StandIn and begin taking messages."""
        timestamp = now_ms()
        http = aiohttp.ClientSession()
        try:
            self._ws = await http.ws_connect(
                self._url,
                headers={
                    TIMESTAMP_HEADER: str(timestamp),
                    SIGNATURE_HEADER: sign_handshake(self._secret, timestamp, "chat"),
                },
                max_msg_size=2 * 1024 * 1024,
            )
        except BaseException:
            await http.close()
            raise
        self._http = http
        self._task = asyncio.ensure_future(self._run())
        logger.info("standin: chat channel open")

    async def aclose(self) -> None:
        self._closed = True
        for task in [self._task, *self._tasks]:
            if task is not None:
                task.cancel()
        pending = [t for t in (self._task, *self._tasks) if t is not None]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._tasks.clear()
        self._chains.clear()
        if self._ws is not None and not self._ws.closed:
            with contextlib.suppress(Exception):
                await self._ws.close()
        if self._http is not None:
            with contextlib.suppress(Exception):
                await self._http.close()

    async def _run(self) -> None:
        assert self._ws is not None
        try:
            async for message in self._ws:
                if message.type is not aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    inbound = parse_inbound(message.data)
                except ValueError as err:
                    logger.warning("standin: dropping malformed chat message: %s", err)
                    continue
                # Dedupe first: StandIn is at-least-once, and a redelivery must
                # not start a second turn for the same activity.
                key = f"{inbound.tenant_id}:{inbound.conversation_id}:{inbound.activity_id}"
                if self._seen.mark_first(key):
                    self._enqueue(inbound)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("standin: chat channel failed")

    def _enqueue(self, message: InboundMessage) -> None:
        chain_key = f"{message.tenant_id}:{message.conversation_id}"
        previous = self._chains.get(chain_key)

        async def run() -> None:
            if previous is not None:
                try:
                    await previous
                except asyncio.CancelledError:
                    # Distinguish the PREVIOUS turn being cancelled from THIS
                    # task being cancelled while parked on it. Swallowing our own
                    # cancellation would run a full turn during shutdown.
                    current = asyncio.current_task()
                    if (
                        current is None
                        or not hasattr(current, "cancelling")
                        or current.cancelling() > 0
                    ):
                        raise
                except Exception:
                    pass  # a failed turn must not dam the chain
            await self._process(message)

        task = asyncio.get_running_loop().create_task(run())
        self._chains[chain_key] = task
        self._tasks.add(task)

        def _done(finished: asyncio.Task[Any]) -> None:
            self._tasks.discard(finished)
            if self._chains.get(chain_key) is finished:
                del self._chains[chain_key]

        task.add_done_callback(_done)

    async def _process(self, message: InboundMessage) -> None:
        # Typing is a courtesy, so it must not sit in FRONT of the turn. Send it
        # and let the agent think; the indicator still lands first.
        await self._send(build_reply(message, "", "typing"))
        try:
            text = await asyncio.wait_for(self._respond(message), timeout=self.TURN_TIMEOUT_S)
            if text and text.strip():
                await self._send(build_reply(message, text))
            else:
                # After a typing indicator, silence looks exactly like a hang.
                logger.warning("standin: chat handler returned an empty answer")
                await self._send(
                    build_reply(
                        message,
                        "I couldn't come up with an answer to that - try rephrasing, "
                        "or ask something else.",
                        "error",
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("standin: chat turn failed")
            await self._send(
                build_reply(
                    message, "Something went wrong answering that - please try again.", "error"
                )
            )

    async def send(
        self,
        *,
        tenant_id: str,
        conversation_id: str,
        text: str,
        idempotency_key: str | None = None,
    ) -> bool:
        """Post into a Teams conversation with no inbound message to answer.

        Useful from inside a call. Best-effort: returns False rather than
        raising, because a failed post must never break a live call.
        """
        payload: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "tenantId": tenant_id,
            "conversationId": conversation_id,
            "kind": "message",
            "text": text,
        }
        if idempotency_key:
            payload["idempotencyKey"] = idempotency_key
        return await self._send(payload)

    async def _send(self, reply: dict[str, Any]) -> bool:
        ws = self._ws
        if ws is None or ws.closed:
            logger.warning("standin: chat channel is not open; dropping a reply")
            return False
        try:
            await ws.send_str(json.dumps(reply, separators=(",", ":")))
            return True
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("standin: chat reply failed", exc_info=True)
            return False


__all__ = ["ChatChannel", "InboundMessage", "SCHEMA_VERSION", "build_reply", "parse_inbound"]
