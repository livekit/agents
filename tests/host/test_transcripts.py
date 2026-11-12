"""A message's readable text and the exported transcript: the agent's expression markup is
stripped, everything else is kept as said."""

from __future__ import annotations

import pytest

from livekit.agents.llm import ChatContext, ChatMessage

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


# Catches: expression tags shown to readers, or stripped from text a person or system wrote.
@pytest.mark.parametrize("role", ["assistant", "user", "system", "developer"])
def test_only_assistant_text_has_its_expression_markup_stripped(role: str) -> None:
    message = ChatMessage(role=role, content=[MARKED])

    assert message.text_content == (CLEAN if role == "assistant" else MARKED)
    assert message.raw_text_content == MARKED


# Catches: a stripped transcript export rewriting the saved history.
def test_a_stripped_export_leaves_the_saved_history_unchanged() -> None:
    context = ChatContext.empty()
    context.add_message(role="user", content=[MARKED])
    context.add_message(role="assistant", content=[MARKED])

    exported = context.to_dict(strip_markup=True)["items"]

    assert [i["content"] for i in exported] == [[MARKED], [CLEAN]]
    assert [i["content"] for i in context.to_dict()["items"]] == [[MARKED], [MARKED]]
