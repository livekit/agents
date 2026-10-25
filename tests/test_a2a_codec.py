"""How a caller reads an answer stream (``a2a/codec.py``): the answer artifact that ended last,
a question from the status message, and a reply streamed before the work went on in the
background, which is not the answer."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from livekit.agents._generated.storage import TaskData
from livekit.agents._generated.storage_run import TaskRecord
from livekit.agents.a2a.codec import (
    StreamEvent,
    TaskUpdate,
    artifact_event,
    final_event,
    submitted_task,
    updates_from_events,
)
from livekit.agents.a2a.extension import Extensions

pytestmark = [pytest.mark.unit]

EXTS = Extensions(livekit=True)


def record(status: str, text: str | None = None, **metadata: object) -> TaskRecord:
    message = (
        {"messageId": "msg_1", "role": "agent", "parts": [{"kind": "text", "text": text}]}
        if text is not None
        else None
    )
    return TaskRecord(
        id="AST_1",
        lane="desk",
        seq=1,
        name=None,
        message_id="m1",
        status=status,
        status_message=message,
        updated_at=0,
        data=TaskData(metadata=dict(metadata)),
        internal=None,
    )


def chunk(text: str, *, artifact_id: str, append: bool, last: bool = False) -> StreamEvent:
    return artifact_event(
        "AST_1",
        "AT_1",
        artifact_id=artifact_id,
        text=text,
        append=append,
        last_chunk=last,
        exts=EXTS,
    )


async def read(events: list[StreamEvent]) -> list[TaskUpdate]:
    async def stream() -> AsyncIterator[StreamEvent]:
        for event in events:
            yield event

    return [u async for u in updates_from_events(stream())]


async def test_the_reply_before_work_in_the_background_is_not_the_answer() -> None:
    updates = await read(
        [
            submitted_task("AST_1", "AT_1"),
            chunk("On it.", artifact_id="art_a", append=False),
            final_event(
                record("working", "On it.", **{"livekit/background": True}), "AT_1", exts=EXTS
            ),
            chunk("Booked for Monday.", artifact_id="art_b", append=False, last=True),
            final_event(record("completed"), "AT_1", exts=EXTS),
        ]
    )
    assert [u.state for u in updates] == ["submitted", "working", "completed"]
    assert updates[1].text == "On it." and updates[1].metadata["livekit/background"] is True
    assert updates[-1].text == "Booked for Monday."


async def test_a_question_is_read_from_the_status_message() -> None:
    updates = await read(
        [
            submitted_task("AST_1", "AT_1"),
            chunk("Let me check.", artifact_id="art_a", append=False),
            final_event(record("input-required", "Which email?"), "AT_1", exts=EXTS),
        ]
    )
    assert updates[-1].state == "input-required" and updates[-1].text == "Which email?"


async def test_the_answer_is_the_artifact_that_ended_last() -> None:
    updates = await read(
        [
            submitted_task("AST_1", "AT_1"),
            chunk("cov", artifact_id="art_stream", append=False),
            chunk("This is not advice.\n\ncovered", artifact_id="art_row", append=False, last=True),
            final_event(record("completed"), "AT_1", exts=EXTS),
        ]
    )
    assert updates[-1].text == "This is not advice.\n\ncovered"
