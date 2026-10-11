from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from livekit.agents import APIConnectionError, llm
from livekit.agents.llm import ChatContext, function_tool
from livekit.plugins import hf
from livekit.plugins.hf import llm as hf_llm
from livekit.plugins.hf.download import models_to_download
from livekit.plugins.hf.tool_parser import ToolCallStreamParser

pytestmark = pytest.mark.unit


def _feed_all(parser: ToolCallStreamParser, chunks: list[str]) -> tuple[str, list[Any]]:
    emitted = "".join(parser.feed(c) for c in chunks)
    rest, calls = parser.flush()
    return emitted + rest, calls


def test_parser_passes_text_through_without_tools() -> None:
    parser = ToolCallStreamParser(detect_tool_calls=False)
    text, calls = _feed_all(parser, ["{not", " a tool}", " <tool_call>"])
    assert text == "{not a tool} <tool_call>"
    assert calls == []


def test_parser_streams_plain_reply() -> None:
    parser = ToolCallStreamParser(detect_tool_calls=True)
    assert parser.feed("Hello") == "Hello"
    assert parser.feed(" there <") == " there "
    assert parser.feed("3") == "<3"
    assert parser.flush() == ("", [])


def test_parser_holds_back_tagged_tool_call() -> None:
    parser = ToolCallStreamParser(detect_tool_calls=True)
    chunks = [
        "Let me check. <tool",
        "_call>\n",
        '{"name": "get_weather", ',
        '"arguments": {"city": ',
    ]
    chunks += ['"Paris"}}\n</tool_call>']
    emitted = [parser.feed(c) for c in chunks]
    assert "".join(emitted) == "Let me check. "
    rest, calls = parser.flush()
    assert rest == ""
    assert [(c.name, json.loads(c.arguments)) for c in calls] == [
        ("get_weather", {"city": "Paris"})
    ]


def test_parser_handles_multiple_and_unterminated_tagged_calls() -> None:
    parser = ToolCallStreamParser(detect_tool_calls=True)
    text, calls = _feed_all(
        parser,
        ['<tool_call>{"name": "a", "arguments": {}}</tool_call>', '<tool_call>{"name": "b"}'],
    )
    assert text == ""
    assert [c.name for c in calls] == ["a", "b"]


def test_parser_parses_bare_json_tool_call() -> None:
    parser = ToolCallStreamParser(detect_tool_calls=True)
    text, calls = _feed_all(parser, ["  {", '"name": "lookup", "parameters": {"id": 3}}'])
    assert text == ""
    assert calls[0].name == "lookup"
    assert json.loads(calls[0].arguments) == {"id": 3}


def test_parser_emits_brace_reply_that_is_not_a_tool_call() -> None:
    parser = ToolCallStreamParser(detect_tool_calls=True)
    text, calls = _feed_all(parser, ["{curly}", " braces"])
    assert text == "{curly} braces"
    assert calls == []


def test_models_to_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIVEKIT_HF_MODELS", raising=False)
    assert models_to_download("default/model") == ["default/model"]
    monkeypatch.setenv("LIVEKIT_HF_MODELS", " a/b, c/d ,")
    assert models_to_download("default/model") == ["a/b", "c/d"]


def test_chat_template_messages_use_dict_arguments() -> None:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="system", content="be brief")
    chat_ctx.add_message(role="user", content="weather in Paris?")
    chat_ctx.insert(
        llm.FunctionCall(call_id="call_1", name="get_weather", arguments='{"city": "Paris"}')
    )
    chat_ctx.insert(
        llm.FunctionCallOutput(call_id="call_1", name="get_weather", output="sunny", is_error=False)
    )

    messages = hf_llm._to_chat_template_messages(chat_ctx)

    tool_call_msgs = [m for m in messages if m.get("tool_calls")]
    assert tool_call_msgs[0]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    assert all(isinstance(m.get("content", ""), str) for m in messages)


class _FakeStreamer:
    """Stands in for TextIteratorStreamer: the fake model pushes text into it."""

    def __init__(self, tokenizer: Any, **kwargs: Any) -> None:
        self._items: list[str | None] = []
        self._cond = threading.Condition()

    def put_text(self, text: str) -> None:
        with self._cond:
            self._items.append(text)
            self._cond.notify()

    def end(self) -> None:
        with self._cond:
            self._items.append(None)
            self._cond.notify()

    def __iter__(self) -> _FakeStreamer:
        return self

    def __next__(self) -> str:
        with self._cond:
            self._cond.wait_for(lambda: bool(self._items))
            item = self._items.pop(0)
        if item is None:
            raise StopIteration
        return item


class _FakeInputs(dict):  # type: ignore[type-arg]
    def to(self, device: Any) -> _FakeInputs:
        return self


class _FakeIds:
    shape = (1, 7)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.template_kwargs: dict[str, Any] = {}

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.template_kwargs = kwargs
        return "prompt"

    def __call__(self, text: str, **kwargs: Any) -> _FakeInputs:
        return _FakeInputs(input_ids=_FakeIds())

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return list(range(len(text.split())))


class _FakeModel:
    device = "cpu"

    def __init__(self, chunks: list[str], error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error

    def generate(self, *, streamer: _FakeStreamer, **kwargs: Any) -> None:
        for chunk in self._chunks:
            streamer.put_text(chunk)
        if self._error is not None:
            raise self._error
        streamer.end()


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> Any:
    import transformers

    monkeypatch.setattr(transformers, "TextIteratorStreamer", _FakeStreamer)

    def _install(model: _FakeModel) -> tuple[hf.LLM, _FakeTokenizer]:
        tokenizer = _FakeTokenizer()
        instance = hf.LLM(model="fake/model")
        loaded = hf_llm._LoadedModel(
            model=model, tokenizer=tokenizer, generate_lock=threading.Lock()
        )
        monkeypatch.setitem(hf_llm._models, instance._opts.key, loaded)
        return instance, tokenizer

    return _install


def _user_ctx() -> ChatContext:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="hi")
    return chat_ctx


async def test_stream_text_and_usage(fake_model: Any) -> None:
    instance, _ = fake_model(_FakeModel(["Hello", " world", "!"]))

    chunks = [c async for c in instance.chat(chat_ctx=_user_ctx())]

    text = "".join(c.delta.content for c in chunks if c.delta and c.delta.content)
    assert text == "Hello world!"
    usage = [c.usage for c in chunks if c.usage]
    assert usage[0] is not None
    assert usage[0].prompt_tokens == 7
    assert usage[0].completion_tokens == 2


async def test_stream_tool_call_is_not_spoken(fake_model: Any) -> None:
    @function_tool
    async def get_weather(city: str) -> str:
        """Get the weather.

        Args:
            city: the city
        """
        return "sunny"

    instance, tokenizer = fake_model(
        _FakeModel(['<tool_call>{"name": "get_weather", ', '"arguments": {"city": "Paris"}}'])
    )

    chunks = [c async for c in instance.chat(chat_ctx=_user_ctx(), tools=[get_weather])]

    assert not any(c.delta and c.delta.content for c in chunks)
    calls = [tc for c in chunks if c.delta and c.delta.tool_calls for tc in c.delta.tool_calls]
    assert [(c.name, json.loads(c.arguments)) for c in calls] == [
        ("get_weather", {"city": "Paris"})
    ]
    assert tokenizer.template_kwargs["tools"][0]["function"]["name"] == "get_weather"


async def test_tool_choice_none_drops_tools(fake_model: Any) -> None:
    @function_tool
    async def noop() -> None:
        """Do nothing."""

    instance, tokenizer = fake_model(_FakeModel(["ok"]))

    _ = [c async for c in instance.chat(chat_ctx=_user_ctx(), tools=[noop], tool_choice="none")]

    assert "tools" not in tokenizer.template_kwargs


async def test_generate_error_does_not_hang(fake_model: Any) -> None:
    instance, _ = fake_model(_FakeModel(["partial"], error=RuntimeError("CUDA out of memory")))

    with pytest.raises(APIConnectionError):
        _ = [c async for c in instance.chat(chat_ctx=_user_ctx())]


def test_warm_up_generates_one_token() -> None:
    calls: list[dict[str, Any]] = []

    class _Model(_FakeModel):
        def generate(self, **kwargs: Any) -> None:  # type: ignore[override]
            calls.append(kwargs)

    hf_llm._warm_up(_Model([]), _FakeTokenizer())

    assert len(calls) == 1
    assert calls[0]["max_new_tokens"] == 1
    assert calls[0]["do_sample"] is False
