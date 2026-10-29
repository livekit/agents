from __future__ import annotations

import pytest

from livekit.agents.llm import ChatContext, ChatMessage

pytestmark = pytest.mark.unit

MARKED = (
    '<expr type="expression" label="happy"/> Press [Enter] to see <b>bold</b>, '
    'read [the docs](https://docs.livekit.io), then 1 < 2. <break time="1s"/> '
    '<expr type="prosody" label="whisper">keep it secret</expr>'
)
CLEAN = (
    "Press [Enter] to see <b>bold</b>, "
    'read [the docs](https://docs.livekit.io), then 1 < 2. <break time="1s"/> '
    "keep it secret"
)


def test_assistant_readable_text_strips_expression_tags_and_preserves_prose() -> None:
    message = ChatMessage(role="assistant", content=[MARKED])

    assert message.text_content == CLEAN
    assert message.raw_text_content == MARKED


@pytest.mark.parametrize("role", ["user", "system", "developer"])
def test_quoted_expression_tags_in_non_assistant_messages_are_preserved(role: str) -> None:
    message = ChatMessage(role=role, content=[MARKED])

    assert message.text_content == MARKED
    assert message.raw_text_content == MARKED


def test_transcript_export_strips_assistant_expressions_without_changing_saved_history() -> None:
    context = ChatContext.empty()
    context.add_message(role="user", content=[MARKED])
    context.add_message(role="assistant", content=[MARKED])

    exported = context.to_dict(strip_markup=True)["items"]

    assert exported[0]["content"] == [MARKED]
    assert exported[1]["content"] == [" " + CLEAN]
    assert context.to_dict()["items"][1]["content"] == [MARKED]
