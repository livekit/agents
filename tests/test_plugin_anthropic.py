"""Unit tests for the Anthropic LLM plugin that do not require a real API key or network."""

from __future__ import annotations

import httpx


def _make_llm(**kwargs):
    from livekit.plugins.anthropic import LLM

    return LLM(api_key="sk-ant-test", **kwargs)


def _filter_thinking(deltas: list[str]) -> str:
    """Stream ``deltas`` through the thinking-tag filter and return the emitted text."""
    from livekit.plugins.anthropic.llm import _ThinkingTagFilter

    fltr = _ThinkingTagFilter()
    out = [fltr.push(text) for text in deltas]
    out.append(fltr.flush())
    return "".join(out)


class TestHttpxTimeoutDefaults:
    def test_default_read_timeout_is_generous(self) -> None:
        """Default read timeout must accommodate adaptive-thinking pauses (≥30 s)."""
        llm = _make_llm()
        read = llm._client._client.timeout.read
        assert read >= 30.0, f"read timeout {read}s is too short for adaptive thinking"

    def test_default_connect_timeout_remains_tight(self) -> None:
        """Connect timeout should stay short so genuine connection failures surface fast."""
        llm = _make_llm()
        connect = llm._client._client.timeout.connect
        assert connect <= 10.0, f"connect timeout {connect}s is unexpectedly long"

    def test_default_timeout_is_split(self) -> None:
        """Default must be an httpx.Timeout object, not a flat scalar."""
        llm = _make_llm()
        t = llm._client._client.timeout
        assert isinstance(t, httpx.Timeout)
        assert t.read != t.connect, "read and connect timeouts should differ in the default"


class TestHttpxTimeoutCustom:
    def test_custom_timeout_honored(self) -> None:
        """A caller-supplied httpx.Timeout is passed through to the httpx client."""
        custom = httpx.Timeout(3.0, read=120.0)
        llm = _make_llm(timeout=custom)
        t = llm._client._client.timeout
        assert t.read == 120.0
        assert t.connect == 3.0

    def test_none_timeout_uses_default(self) -> None:
        """Passing timeout=None must fall back to the built-in default."""
        llm = _make_llm(timeout=None)
        assert llm._client._client.timeout.read >= 30.0

    def test_explicit_client_bypasses_timeout_param(self) -> None:
        """When a pre-built client= is supplied, timeout= is ignored (client wins)."""
        import anthropic

        tight_client = anthropic.AsyncClient(
            api_key="sk-ant-test",
            http_client=httpx.AsyncClient(timeout=httpx.Timeout(1.0)),
        )
        # timeout= argument should have no effect here
        llm = _make_llm(client=tight_client, timeout=httpx.Timeout(5.0, read=999.0))
        assert llm._client._client.timeout.read == 1.0


class TestThinkingTagStripping:
    """When tools are attached, Claude may wrap chain-of-thought in <thinking> tags.

    Those tags must be stripped from the streamed text, but the actual assistant
    response (the part that feeds TTS) must always be emitted — even when the tags
    are split across streaming deltas, which is the normal token-by-token case.
    """

    def test_plain_answer_passes_through(self) -> None:
        assert _filter_thinking(["Hello", " there", "!"]) == "Hello there!"

    def test_thinking_block_stripped_tags_on_own_deltas(self) -> None:
        got = _filter_thinking(["<thinking>", "the user said hi", "</thinking>", "Hello there!"])
        assert got == "Hello there!"

    def test_answer_emitted_when_closing_tag_split_across_deltas(self) -> None:
        # regression: the closing tag streamed as "</" + "thinking>" used to leave the
        # parser permanently in "ignoring" mode, dropping the entire spoken answer so
        # no audio was ever published.
        got = _filter_thinking(["<thinking>", "reasoning", "</", "thinking>", "Hello there!"])
        assert got == "Hello there!"

    def test_answer_emitted_when_opening_tag_split_across_deltas(self) -> None:
        got = _filter_thinking(["<", "thinking>", "reasoning", "</thinking>", "Hello there!"])
        assert got == "Hello there!"

    def test_thinking_block_with_leading_whitespace(self) -> None:
        got = _filter_thinking(["\n<thinking>", "reasoning", "</thinking>", "Hi!"])
        assert got == "\nHi!"

    def test_text_resembling_tag_is_not_dropped(self) -> None:
        # text that merely looks like the start of a tag must not be silently swallowed
        assert _filter_thinking(["1 < 2 is true"]) == "1 < 2 is true"

    def test_partial_opening_tag_at_end_is_flushed(self) -> None:
        # a dangling "<th" that never completes into a real tag must not be lost
        assert _filter_thinking(["all good", "<th"]) == "all good<th"


class TestParseEventThinkingIntegration:
    """End-to-end check that ``_parse_event`` emits the spoken answer with tools attached."""

    async def test_split_closing_tag_still_emits_answer(self) -> None:
        import anthropic.types as at

        from livekit.agents.llm.chat_context import ChatContext
        from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
        from livekit.plugins.anthropic.llm import LLMStream

        async def _never() -> None:
            return None

        never = _never()
        stream = LLMStream(
            _make_llm(),
            anthropic_stream=never,
            chat_ctx=ChatContext.empty(),
            tools=["smoke"],
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
        )
        # we only drive _parse_event manually; tear down the background _run task
        await stream.aclose()
        never.close()  # the cancelled _run never awaited it

        events = [
            at.RawContentBlockStartEvent(
                type="content_block_start",
                index=0,
                content_block=at.TextBlock(type="text", text="", citations=None),
            ),
            at.RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=at.TextDelta(type="text_delta", text="<thinking>"),
            ),
            at.RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=at.TextDelta(type="text_delta", text="the user said hi"),
            ),
            at.RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=at.TextDelta(type="text_delta", text="</"),
            ),
            at.RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=at.TextDelta(type="text_delta", text="thinking>"),
            ),
            at.RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=at.TextDelta(type="text_delta", text="Hello there!"),
            ),
            at.RawContentBlockStopEvent(type="content_block_stop", index=0),
        ]

        emitted = ""
        for ev in events:
            chunk = stream._parse_event(ev)
            if chunk is not None and chunk.delta is not None and chunk.delta.content:
                emitted += chunk.delta.content

        assert emitted == "Hello there!"
