"""Text job handling: TextMessageContext, request parsing, and entrypoint."""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING

from aiohttp import web
from google.protobuf.json_format import MessageToDict, ParseDict

from livekit.protocol.agent_pb import agent_text

from . import utils
from ._exceptions import TextMessageError
from .log import logger

if TYPE_CHECKING:
    from .job import JobContext
    from .voice.agent_session import _AgentSessionState
    from .voice.run_result import RunEvent
    from .worker import AgentServer


class TextMessageContext:
    def __init__(self, *, job_ctx: JobContext, text_request: agent_text.TextMessageRequest) -> None:
        self._job_ctx = job_ctx
        self._text_request = text_request

        # read session snapshot
        if text_request.HasField("session_state"):
            assert text_request.session_state.WhichOneof("data") == "snapshot", (
                "session state should be resolved before starting the job"
            )
            self._session_snapshot: bytes | None = text_request.session_state.snapshot
        else:
            self._session_snapshot = None
        self._session_state: _AgentSessionState | None = None

    async def send_response(self, ev: RunEvent) -> None:
        from . import llm
        from .ipc import proto

        if ev.item.type not in [
            "message",
            "function_call",
            "function_call_output",
            "agent_handoff",
        ]:
            return

        msg = proto.TextResponse(session_id=self.session_id)
        item_pb = llm.chat_context.chat_item_to_proto(ev.item)
        msg.event = agent_text.TextMessageResponse(
            session_id=self.session_id,
            message_id=self._text_request.message_id,
            **{str(ev.item.type): getattr(item_pb, ev.item.type)},
        )
        await self._job_ctx._ipc_client.send(msg)

    @property
    def session_id(self) -> str:
        return self._text_request.session_id

    @property
    def session_state(self) -> _AgentSessionState | None:
        from .utils.session_store import SessionStore

        try:
            if self._session_state is None and self._session_snapshot:
                with SessionStore(self._session_snapshot) as store:
                    self._session_state = store.export_state()
        except Exception as e:
            logger.exception(
                "failed to export session state from snapshot",
                extra={"session_id": self.session_id},
            )
            raise TextMessageError(
                f"failed to export session state from snapshot with error: {str(e)}",
                code=agent_text.INTERNAL_ERROR,
            ) from e

        return self._session_state

    @property
    def text(self) -> str:
        return self._text_request.text

    async def _complete(self, exc: TextMessageError | None = None) -> None:
        from .ipc import proto
        from .utils.session_store import SessionStore

        msg = proto.TextResponse(session_id=self.session_id)
        msg.event = agent_text.TextMessageResponse(
            session_id=self.session_id,
            message_id=self._text_request.message_id,
            complete=agent_text.TextMessageComplete(),
        )
        if exc:
            msg.event.complete.error.CopyFrom(exc.to_proto())
            await self._job_ctx._ipc_client.send(msg)
            return

        session = self._job_ctx._primary_agent_session
        if not session:
            logger.error(
                "no primary agent session found", extra={"text_session_id": self.session_id}
            )
            msg.event.complete.error.CopyFrom(
                TextMessageError("no primary agent session found").to_proto()
            )
            await self._job_ctx._ipc_client.send(msg)
            return

        with SessionStore(self._session_snapshot) as old_store:
            new_version = old_store.version + 1
            session_state = msg.event.complete.session_state
            session_state.version = new_version
            with SessionStore.from_state(session.get_state(), version=new_version) as new_store:
                if not self._session_snapshot:
                    session_state.snapshot = new_store.export_snapshot()
                else:
                    session_state.delta = old_store.compute_delta(new_store)

        await self._job_ctx._ipc_client.send(msg)


async def _text_job_entrypoint(
    handler_fnc: Callable[[TextMessageContext], Awaitable[None]],
    ctx: JobContext,
) -> None:
    """Module-level text job entrypoint (picklable via functools.partial)."""
    assert ctx.text_message_context is not None

    exc: TextMessageError | None = None
    try:
        await handler_fnc(ctx.text_message_context)
    except TextMessageError as e:
        exc = e
    except Exception as e:
        exc = TextMessageError(
            f"error in text handler: {str(e)}",
            code=agent_text.TEXT_HANDLER_ERROR,
        )
        logger.exception(
            "error in text handler",
            extra={"session_id": ctx.text_message_context.session_id},
        )
    finally:
        try:
            await ctx.text_message_context._complete(exc)
        except Exception as e:
            logger.exception(
                "error completing text session",
                extra={"session_id": ctx.text_message_context.session_id},
            )
            if exc is None:
                with contextlib.suppress(Exception):
                    await ctx.text_message_context._complete(
                        TextMessageError(f"error completing: {str(e)}")
                    )

        if session := ctx._primary_agent_session:
            with contextlib.suppress(Exception):
                session._stop_durable_scheduler()
                await session.aclose()
        ctx.shutdown()


async def _handle_text_request(
    text: str,
    endpoint: str = "",
    session_id: str | None = None,
    message_id: str | None = None,
    agent_name: str | None = None,
    metadata: dict | None = None,
    session_state: dict | None = None,
    *,
    agent_server: AgentServer,
) -> AsyncIterator[str]:
    """Handle POST /text/{endpoint}[/sessions/{session_id}]"""
    logger.info(f"handling text request: endpoint={endpoint} session_id={session_id}")

    handler_fnc = agent_server._text_handler_fncs.get(endpoint)
    if handler_fnc is None:
        raise web.HTTPNotFound(
            reason=json.dumps({"error": f"Text service for '{endpoint}' not found"}),
            content_type="application/json",
        )

    # parse TextMessageRequest from args
    body: dict[str, object] = {"text": text}
    body["message_id"] = message_id or utils.shortuuid("text_msg_")
    body["agent_name"] = agent_name or agent_server._agent_name
    body["session_id"] = session_id or utils.shortuuid("text_session_")
    if metadata is not None:
        body["metadata"] = metadata
    if session_state is not None:
        body["session_state"] = session_state

    try:
        text_req = ParseDict(body, agent_text.TextMessageRequest(), ignore_unknown_fields=True)
    except Exception as e:
        raise web.HTTPBadRequest(
            reason=json.dumps({"error": f"Invalid request body: {str(e)}"}),
            content_type="application/json",
        ) from e

    async def _stream() -> AsyncIterator[str]:
        completed = False

        def _format_message(msg: agent_text.TextMessageResponse) -> str:
            msg_json = json.dumps(MessageToDict(msg, preserving_proto_field_name=True))
            return msg_json + "\n"

        try:
            session_info = await agent_server._launch_text_job(endpoint, text_req, handler_fnc)

            async for ev in session_info.event_ch:
                if completed:
                    logger.warning("received message after completion")
                else:
                    yield _format_message(ev)
                    if ev.WhichOneof("event") == "complete":
                        completed = True

        except TextMessageError as e:
            logger.error(
                "error processing text request",
                extra={
                    "session_id": text_req.session_id,
                    "error": e.message,
                    "error_code": agent_text.TextMessageErrorCode.Name(e.code),
                },
            )
            if not completed:
                yield _format_message(
                    agent_text.TextMessageResponse(
                        session_id=text_req.session_id,
                        message_id=text_req.message_id,
                        complete=agent_text.TextMessageComplete(error=e.to_proto()),
                    )
                )
        except Exception:
            logger.exception(
                "unexpected error processing text request",
                extra={"session_id": text_req.session_id},
            )
            if not completed:
                yield _format_message(
                    agent_text.TextMessageResponse(
                        session_id=text_req.session_id,
                        message_id=text_req.message_id,
                        complete=agent_text.TextMessageComplete(
                            error=TextMessageError("internal error").to_proto()
                        ),
                    )
                )

    return _stream()
