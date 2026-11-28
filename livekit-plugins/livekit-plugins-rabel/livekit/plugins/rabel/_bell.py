# Copyright 2026 Rabel Labs
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

"""One conversation with the Rabel API (`WS /v1/conversation`), kept open for a speech stream.

Hello: the first message carries `key`, `type` ("hello"), `v` (3), `agent`, `host` and `learn`, and
`min_chance` only when the developer set one. The API answers with a `ready` that carries `"v": 3`.
Any other `ready` ends the conversation and the speech stream runs as a plain pass-through for the
rest of the call.
In: `{"type": "heard", "text"}` with the turn's words so far, `{"type": "end_of_turn", "text"}` with
its final words and, unless the report is off, `guess` ("none", "kept" or "dropped") and with "kept"
`head_start_ms`; on a turn with more than one guess, `guess` is what happened to the last and
`dropped_before` counts the ones before it, each dropped.
Out: `ready` (`v`, `min_chance`, null when the developer set none, `hold_line`, `hold_always` and
`hold_updates`), `guess` (`text`, `heard`, `chance`), `end_of_turn`, `error` (`status`, `error`,
`message`, `retry_after_s`). Fields this client does not know are ignored.

Nothing here ever raises into the caller. Words are sent only while the API is ready; before that,
and after any failure, they are dropped and the speech stream runs as a plain pass-through. Each
kind of failure is logged once.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable
from typing import Any

import aiohttp

from livekit.agents import utils

from .log import logger

# The reopening rules (the Rabel API's docs).
# A refusal that will not clear by itself during a call: no new conversation is opened for this
# stream (400 bad_message, 401 bad_key, 402 no_seconds, 404 unknown_agent, 409 read_only_agent).
_STOP_STATUSES = {400, 401, 402, 404, 409}
# Anything else that ends a conversation (408, 429, 503, 529, a dropped connection, a timeout) is
# retried after these waits in turn, the last one repeating; the API's retry_after_s is used instead
# when it is longer, up to the cap.
_BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
_MAX_RETRY_AFTER_S = 300.0
# A conversation that was open this long and then closed cleanly (the API closes one after an hour)
# reopens at once.
_LONG_CONVERSATION_S = 60.0
# The back-off starts again only after a conversation that stayed up this long after its ready, so
# an API that sends ready and then closes is reopened on the growing waits, not every second.
_STEADY_S = 30.0
# One timeout from opening the connection until the API's ready; past it the attempt counts as a
# timeout and is retried.
_OPEN_TIMEOUT_S = 2.0
# How long closing the WebSocket waits for the API's close frame.
_WS_CLOSE_S = 2.0
# Bell sends nothing while it has no guess, so a quiet line is normal. The connection is watched
# with LiveKit's WebSocketMonitor: after a third of this with nothing received it pings, and if no
# pong or message comes back within this time the connection counts as dead (a half-open socket)
# and the conversation is reopened on the back-off. The Rabel API answers pings in its WebSocket
# server, not in the conversation's code, so a healthy quiet call always answers in time.
_LIVENESS_TIMEOUT_S = 15.0


def _ws_timeout() -> Any:
    """The close wait as the installed aiohttp takes it: a ClientWSTimeout from aiohttp 3.11, a
    number of seconds before (livekit-agents allows aiohttp 3.10)."""
    ws_timeout = getattr(aiohttp, "ClientWSTimeout", None)
    return _WS_CLOSE_S if ws_timeout is None else ws_timeout(ws_close=_WS_CLOSE_S)


def _backoff(attempt: int) -> float:
    return _BACKOFF_S[min(attempt, len(_BACKOFF_S) - 1)]


class BellConnection:
    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        hello: dict[str, Any],
        on_guess: Callable[[dict[str, Any]], None],
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._hello = hello
        self._on_guess = on_guess
        self._own_session = http_session is None
        self._session = http_session
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._ready = False
        self._hold_line: float | None = None
        self._hold_always: bool | None = None
        self._hold_updates: int | None = None
        self._min_chance: float | None = None
        self._wrong_version = False  # the API answered with a `ready` that is not version 3
        self._conversation = 0  # counts the conversations that reached ready
        self._closed = False
        self._logged: set[str] = set()
        self._out: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self.stats: dict[str, int] = {
            "sent": 0,
            "dropped_not_ready": 0,
            "guesses": 0,
            "connects": 0,
        }

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def conversation(self) -> int:
        """Which conversation is open.

        A new number each time one reaches ready, 0 before the first.
        """
        return self._conversation

    @property
    def hold_line(self) -> float | None:
        """Rule 3's line from the API's `ready`, or None when the API did not send a usable one."""
        return self._hold_line

    @property
    def hold_always(self) -> bool | None:
        """Rule 3 for any last word, whatever its confidence, from the API's `ready`; None when the
        API did not send it or sent a value that is not true or false."""
        return self._hold_always

    @property
    def hold_updates(self) -> int | None:
        """How many updates in a row rule 3 may hold an interim (1 or 2), from the API's `ready`;
        None when the API did not send it or sent a value out of range."""
        return self._hold_updates

    @property
    def min_chance(self) -> float | None:
        """The min_chance in force, from the API's `ready`, or None when it sent none (the developer
        set none: Bell's balanced setting)."""
        return self._min_chance

    def log_once(self, kind: str, line: str) -> None:
        """One logged line for each kind of failure on this stream."""
        self._log_once(kind, line)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="rabel.BellConnection")

    def heard(self, text: str) -> None:
        """The turn's words so far, and nothing else."""
        self._send({"type": "heard", "text": text})

    def end_of_turn(
        self,
        text: str,
        guess: str | None = None,
        head_start_ms: int | None = None,
        dropped_before: int = 0,
    ) -> None:
        """The turn's final words, and when given, the report of what happened to Bell's guess in
        the turn (the Rabel API's docs): `guess` "none", "kept" or "dropped", and with "kept" the
        head start in ms. With more than one guess in the turn, `guess` is the last one's and
        `dropped_before` the number of guesses dropped before it (sent only when there were any, so
        a turn with one guess reports as before)."""
        msg: dict[str, Any] = {"type": "end_of_turn", "text": text}
        if guess is not None:
            msg["guess"] = guess
            if guess == "kept" and head_start_ms is not None:
                msg["head_start_ms"] = head_start_ms
            if dropped_before > 0:
                msg["dropped_before"] = dropped_before
        self._send(msg)

    def _send(self, msg: dict[str, Any]) -> None:
        if not self._ready or self._closed:
            self.stats["dropped_not_ready"] += 1
            return
        try:
            self._out.put_nowait(json.dumps(msg))
        except Exception as e:  # noqa: BLE001
            self._log_once(
                "encode",
                f"Bell: a message could not be sent ({e!r}); the speech-to-text passes through",
            )

    def _log_once(self, kind: str, line: str) -> None:
        if kind not in self._logged:
            self._logged.add(kind)
            logger.warning(line)

    async def aclose(self) -> None:
        self._closed = True
        self._ready = False
        self._wake.set()
        if self._task is not None:
            await utils.aio.cancel_and_wait(self._task)
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        if self._own_session and self._session is not None:
            with contextlib.suppress(Exception):
                await self._session.close()

    async def _run(self) -> None:
        attempt = 0  # failed attempts since the last conversation that stayed up
        while not self._closed:
            opened_at: float | None = None
            refusal: dict[str, Any] | None = None
            clean = False
            try:
                opened_at, refusal, clean = await self._converse()
            except asyncio.CancelledError:
                raise
            except (asyncio.TimeoutError, TimeoutError):
                self._log_once(
                    "timeout",
                    f"Bell: the Rabel API was not ready within {_OPEN_TIMEOUT_S} s; "
                    "the speech-to-text passes through unchanged until it is",
                )
            except Exception as e:  # noqa: BLE001 -- down, unreachable, a broken socket
                self._log_once(
                    "down",
                    f"Bell: the Rabel API is not reachable ({type(e).__name__}: {e}); "
                    "the speech-to-text passes through unchanged",
                )
            finally:
                self._ready = False
                self._ws = None
                while not self._out.empty():
                    self._out.get_nowait()
            if self._closed or self._wrong_version:
                # an API that does not speak version 3: the stream passes through for the
                # rest of the call
                return
            if opened_at is not None and time.perf_counter() - opened_at >= _STEADY_S:
                attempt = 0
            wait = self._wait_after(attempt, opened_at, refusal, clean)
            if wait is None:
                return  # refused for good: the stream passes through for the rest of the call
            if wait > 0:
                attempt += 1
                self._wake.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), wait)

    def _wait_after(
        self,
        attempt: int,
        opened_at: float | None,
        refusal: dict[str, Any] | None,
        clean: bool,
    ) -> float | None:
        """The wait before the next conversation, by the API's reopening rules.

        None: open no other.
        """
        if refusal is not None:
            status = refusal.get("status")
            if status in _STOP_STATUSES:
                return None
            wait = _backoff(attempt)
            retry_after = refusal.get("retry_after_s")
            if (
                isinstance(retry_after, (int, float))
                and not isinstance(retry_after, bool)
                and retry_after > wait
            ):
                wait = min(float(retry_after), _MAX_RETRY_AFTER_S)
            return wait
        if (
            opened_at is not None
            and clean
            and time.perf_counter() - opened_at >= _LONG_CONVERSATION_S
        ):
            return 0.0
        if opened_at is not None:
            self._log_once(
                "closed",
                "Bell: the Rabel API closed the conversation early; "
                "the speech-to-text passes through unchanged until it reopens",
            )
        return _backoff(attempt)

    def _log_refusal(self, m: dict[str, Any]) -> None:
        status, err = m.get("status"), m.get("error")
        self._log_once(
            f"refused:{status}:{err}",
            f"Bell: the Rabel API refused ({status} {err}: {m.get('message')}); "
            "the speech-to-text passes through unchanged",
        )

    async def _converse(self) -> tuple[float | None, dict[str, Any] | None, bool]:
        """One conversation.

        Returns (when it reached ready or None, the API's error message if any, closed cleanly).
        """
        if self._session is None:
            try:
                # in a LiveKit job, the job's shared session, as LiveKit's own plugins use it
                self._session = utils.http_context.http_session()
                self._own_session = False
            except RuntimeError:
                # outside a job: one of its own, closed with it
                self._session = aiohttp.ClientSession()
        ws_url = (
            self._url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
            + "/v1/conversation"
        )
        self.stats["connects"] += 1
        deadline = time.perf_counter() + _OPEN_TIMEOUT_S
        ws = await asyncio.wait_for(
            self._session.ws_connect(
                ws_url,
                timeout=_ws_timeout(),
                heartbeat=None,
                autoping=False,
                max_msg_size=1 << 20,
                autoclose=True,
            ),
            _OPEN_TIMEOUT_S,
        )
        self._ws = ws
        # Every read goes through the monitor: it answers the API's pings, pings a quiet line and
        # raises APIConnectionError when the line stays dead, which `_run` handles as "down".
        monitor = utils.aio.WebSocketMonitor(ws, liveness_timeout=_LIVENESS_TIMEOUT_S)
        sender: asyncio.Task[None] | None = None
        opened_at: float | None = None
        refusal: dict[str, Any] | None = None
        clean = False
        try:
            await ws.send_str(json.dumps({"key": self._api_key, **self._hello}))
            while True:
                if opened_at is None:
                    # until ready, the open timeout runs; a timeout ends this attempt
                    # (retried on the back-off)
                    msg = await asyncio.wait_for(
                        monitor.receive(), max(deadline - time.perf_counter(), 0.0)
                    )
                else:
                    msg = await monitor.receive()
                if msg.type != aiohttp.WSMsgType.TEXT:
                    break
                try:
                    m = json.loads(msg.data)
                except ValueError:
                    self._log_once(
                        "wrong_data", "Bell: the Rabel API sent a message that is not JSON; ignored"
                    )
                    continue
                kind = m.get("type") if isinstance(m, dict) else None
                if kind == "ready" and opened_at is None:
                    v = m.get("v")
                    if v != 3 or isinstance(v, bool):
                        self._wrong_version = True
                        self._log_once(
                            "version",
                            f"Bell: the Rabel API answered with version {v!r}, not 3; "
                            "the speech-to-text passes through unchanged",
                        )
                        break
                    self._hold_line = _unit_number(m.get("hold_line"))
                    self._hold_always = self._setting(
                        m, "hold_always", _true_or_false, "true or false"
                    )
                    self._hold_updates = self._setting(m, "hold_updates", _hold_updates, "1 or 2")
                    self._min_chance = _unit_number(m.get("min_chance"))
                    self._conversation += 1
                    opened_at = time.perf_counter()
                    self._ready = True
                    sender = asyncio.create_task(self._send_loop(ws))
                elif kind == "guess" and opened_at is not None:
                    self.stats["guesses"] += 1
                    try:
                        self._on_guess(m)
                    except Exception as e:  # noqa: BLE001
                        self._log_once(
                            "guess",
                            f"Bell: a guess could not be handled ({e!r}); "
                            "the speech-to-text passes through",
                        )
                elif kind == "error":
                    refusal = m
                    self._log_refusal(m)
                # other message types (end_of_turn, ones this version does not know) are ignored
            clean = ws.close_code == aiohttp.WSCloseCode.OK
            return opened_at, refusal, clean
        finally:
            self._ready = False
            if sender is not None:
                await utils.aio.cancel_and_wait(sender)
            with contextlib.suppress(Exception):
                await ws.close()

    def _setting(self, m: dict[str, Any], name: str, read: Callable[[Any], Any], want: str) -> Any:
        """A rule-3 setting from `ready`: None when absent; None and one logged line when it is not
        a value this client takes, so the default is used."""
        v = m.get(name)
        if v is None:
            return None
        got = read(v)
        if got is None:
            self._log_once(
                f"bad_{name}",
                f"Bell: the Rabel API sent {name} {v!r}, not {want}; the default is used",
            )
        return got

    async def _send_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            data = await self._out.get()
            await ws.send_str(data)
            self.stats["sent"] += 1


def _unit_number(v: Any) -> float | None:
    """A number in [0, 1] from the API, else None (missing or malformed)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
        return None
    return float(v)


def _true_or_false(v: Any) -> bool | None:
    return v if isinstance(v, bool) else None


def _hold_updates(v: Any) -> int | None:
    """1 or 2 (an int, or a float with that value), else None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v not in (1, 2):
        return None
    return int(v)
