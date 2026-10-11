from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from livekit.agents import APIConnectionError, APIConnectOptions, llm
from livekit.agents.llm import ChatContext, function_tool
from livekit.plugins import hf
from livekit.plugins.hf import llm as hf_llm
from livekit.plugins.hf.download import models_to_download
from livekit.plugins.hf.tool_parser import ToolCallStreamParser

pytestmark = pytest.mark.unit

_TOOLS = ["get_weather", "lookup", "a", "b"]


def _feed_all(parser: ToolCallStreamParser, chunks: list[str]) -> tuple[str, list[Any]]:
    emitted = "".join(parser.feed(c) for c in chunks)
    rest, calls = parser.flush()
    return emitted + rest, calls


def test_parser_passes_text_through_without_tools() -> None:
    parser = ToolCallStreamParser(tool_names=[])
    text, calls = _feed_all(parser, ["{not", " a tool}", " <tool_call>"])
    assert text == "{not a tool} <tool_call>"
    assert calls == []


def test_parser_streams_plain_reply() -> None:
    parser = ToolCallStreamParser(tool_names=_TOOLS)
    assert parser.feed("Hello") == "Hello"
    assert parser.feed(" there <") == " there "
    assert parser.feed("3") == "<3"
    assert parser.flush() == ("", [])


def test_parser_holds_back_tagged_tool_call() -> None:
    parser = ToolCallStreamParser(tool_names=_TOOLS)
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
    parser = ToolCallStreamParser(tool_names=_TOOLS)
    text, calls = _feed_all(
        parser,
        ['<tool_call>{"name": "a", "arguments": {}}</tool_call>', '<tool_call>{"name": "b"}'],
    )
    assert text == ""
    assert [c.name for c in calls] == ["a", "b"]


def test_parser_parses_bare_json_tool_call() -> None:
    parser = ToolCallStreamParser(tool_names=_TOOLS)
    text, calls = _feed_all(parser, ["  {", '"name": "lookup", "parameters": {"id": 3}}'])
    assert text == ""
    assert calls[0].name == "lookup"
    assert json.loads(calls[0].arguments) == {"id": 3}


def test_parser_keeps_text_after_tool_call() -> None:
    parser = ToolCallStreamParser(tool_names=_TOOLS)
    text, calls = _feed_all(
        parser,
        ["Checking. <tool_call>", '{"name": "lookup", "arguments": {}}', "</tool_call> Found it."],
    )
    assert text == "Checking.  Found it."
    assert [c.name for c in calls] == ["lookup"]


def test_parser_ignores_json_answer_naming_no_tool() -> None:
    parser = ToolCallStreamParser(tool_names=_TOOLS)
    text, calls = _feed_all(parser, ['{"name": "Alice", "age": 3}'])
    assert text == '{"name": "Alice", "age": 3}'
    assert calls == []


def test_parser_bare_json_needs_arguments() -> None:
    parser = ToolCallStreamParser(tool_names=_TOOLS)
    text, calls = _feed_all(parser, ['{"name": "lookup"}'])
    assert text == '{"name": "lookup"}'
    assert calls == []


def test_parser_drops_tagged_call_to_unknown_tool() -> None:
    parser = ToolCallStreamParser(tool_names=_TOOLS)
    text, calls = _feed_all(parser, ['<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>'])
    assert text == ""
    assert calls == []


def test_parser_holds_content_until_flush() -> None:
    parser = ToolCallStreamParser(tool_names=_TOOLS, hold_content=True)
    assert parser.feed("Sure. ") == ""
    assert parser.feed('<tool_call>{"name": "a", "arguments": {}}</tool_call>') == ""
    text, calls = parser.flush()
    assert text == "Sure. "
    assert [c.name for c in calls] == ["a"]


def test_parser_emits_brace_reply_that_is_not_a_tool_call() -> None:
    parser = ToolCallStreamParser(tool_names=_TOOLS)
    text, calls = _feed_all(parser, ["{curly}", " braces"])
    assert text == "{curly} braces"
    assert calls == []


def test_models_to_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIVEKIT_HF_MODELS", raising=False)
    assert models_to_download("default/model") == [("default/model", None)]
    monkeypatch.setenv("LIVEKIT_HF_MODELS", " a/b, c/d ,")
    assert models_to_download("default/model") == [("a/b", None), ("c/d", None)]
    monkeypatch.setenv("LIVEKIT_HF_MODELS", "a/b@abc123")
    assert models_to_download("default/model") == [("a/b", "abc123")]
    monkeypatch.setenv("LIVEKIT_HF_MODELS", "gpt2@abc123, gpt2")
    assert models_to_download("default/model") == [("gpt2", "abc123"), ("gpt2", None)]
    monkeypatch.setenv("LIVEKIT_HF_MODELS", "a/b@v1.0, c/d, ./local@dir, e/f@")
    assert models_to_download("default/model") == [
        ("a/b", "v1.0"),
        ("c/d", None),
        ("./local@dir", None),
        ("e/f@", None),
    ]


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
    def __init__(self, chat_template: str = "") -> None:
        self.chat_template = chat_template
        self.template_kwargs: dict[str, Any] = {}
        self.messages: list[dict[str, Any]] = []
        self.prompt = ""

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.template_kwargs = kwargs
        self.messages = messages
        return "prompt"

    def __call__(self, text: str, **kwargs: Any) -> _FakeInputs:
        self.prompt = text
        return _FakeInputs(input_ids=_FakeIds())

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return list(range(len(text.split())))


class _FakeModel:
    device = "cpu"

    def __init__(
        self,
        chunks: list[str],
        error: Exception | None = None,
        generated_ids: int | None = None,
    ) -> None:
        self._chunks = chunks
        self._error = error
        self._generated_ids = generated_ids

    def generate(self, *, streamer: _FakeStreamer, **kwargs: Any) -> Any:
        for chunk in self._chunks:
            streamer.put_text(chunk)
        if self._error is not None:
            raise self._error
        streamer.end()
        if self._generated_ids is None:
            return None
        return _FakeSequences(_FakeIds.shape[1] + self._generated_ids)


class _FakeSequences:
    def __init__(self, length: int) -> None:
        self.shape = (1, length)


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> Any:
    import transformers

    monkeypatch.setattr(transformers, "TextIteratorStreamer", _FakeStreamer)

    def _install(model: _FakeModel, chat_template: str = "") -> tuple[hf.LLM, _FakeTokenizer]:
        tokenizer = _FakeTokenizer(chat_template)
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


@function_tool
async def get_weather(city: str) -> str:
    """Get the weather.

    Args:
        city: the city
    """
    return "sunny"


@function_tool
async def get_time() -> str:
    """Get the time."""
    return "noon"


async def test_usage_counts_generated_token_ids(fake_model: Any) -> None:
    instance, _ = fake_model(_FakeModel(["Hi", "!"], generated_ids=5))

    chunks = [c async for c in instance.chat(chat_ctx=_user_ctx())]

    usage = [c.usage for c in chunks if c.usage][0]
    assert usage is not None
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 12


async def test_required_tool_prefills_tagged_call(fake_model: Any) -> None:
    instance, tokenizer = fake_model(
        _FakeModel(['{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>']),
        chat_template="... <tool_call> ...",
    )

    chunks = [
        c
        async for c in instance.chat(
            chat_ctx=_user_ctx(), tools=[get_weather, get_time], tool_choice="required"
        )
    ]

    assert tokenizer.prompt == "prompt<tool_call>\n"
    calls = [tc for c in chunks if c.delta and c.delta.tool_calls for tc in c.delta.tool_calls]
    assert [(c.name, json.loads(c.arguments)) for c in calls] == [
        ("get_weather", {"city": "Paris"})
    ]


async def test_named_tool_choice_offers_only_that_tool(fake_model: Any) -> None:
    instance, tokenizer = fake_model(
        _FakeModel(['{"city": "Paris"}}</tool_call>']), chat_template="<tool_call>"
    )

    chunks = [
        c
        async for c in instance.chat(
            chat_ctx=_user_ctx(),
            tools=[get_weather, get_time],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
    ]

    assert [t["function"]["name"] for t in tokenizer.template_kwargs["tools"]] == ["get_weather"]
    assert tokenizer.prompt.endswith('<tool_call>\n{"name": "get_weather", "arguments": ')
    calls = [tc for c in chunks if c.delta and c.delta.tool_calls for tc in c.delta.tool_calls]
    assert [c.name for c in calls] == ["get_weather"]


async def test_required_tool_without_call_fails(fake_model: Any) -> None:
    instance, tokenizer = fake_model(_FakeModel(["It is sunny."]))

    stream = instance.chat(
        chat_ctx=_user_ctx(),
        tools=[get_weather],
        tool_choice="required",
        conn_options=APIConnectOptions(max_retry=0),
    )
    with pytest.raises(APIConnectionError):
        _ = [c async for c in stream]

    # templates without a call tag get an instruction instead of a prefill
    assert tokenizer.messages[0]["role"] == "system"
    assert "must reply by calling" in tokenizer.messages[0]["content"]


def test_loading_one_model_does_not_block_another(monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()
    started = threading.Event()

    def _slow_load(key: Any, *, local_files_only: bool) -> Any:
        started.set()
        release.wait(5)
        return hf_llm._LoadedModel(model=None, tokenizer=None, generate_lock=threading.Lock())

    monkeypatch.setattr(hf_llm, "_load_model", _slow_load)
    slow, cached = hf.LLM(model="slow/model"), hf.LLM(model="cached/model")
    cached_model = hf_llm._LoadedModel(model=None, tokenizer=None, generate_lock=threading.Lock())
    monkeypatch.setitem(hf_llm._models, cached._opts.key, cached_model)
    monkeypatch.setattr(hf_llm, "_load_locks", {})

    loader = threading.Thread(target=slow.load)
    loader.start()
    try:
        assert started.wait(5)
        assert cached._get_model() is cached_model
    finally:
        release.set()
        loader.join(5)
        hf_llm._models.pop(slow._opts.key, None)


async def test_parallel_tool_calls_false_keeps_first(fake_model: Any) -> None:
    call = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    other = '<tool_call>{"name": "get_time", "arguments": {}}</tool_call>'

    async def _names(**kwargs: Any) -> list[str]:
        instance, _ = fake_model(_FakeModel([call, call, other]))
        chunks = [
            c
            async for c in instance.chat(
                chat_ctx=_user_ctx(), tools=[get_weather, get_time], **kwargs
            )
        ]
        return [
            tc.name for c in chunks if c.delta and c.delta.tool_calls for tc in c.delta.tool_calls
        ]

    assert await _names() == ["get_weather", "get_weather", "get_time"]
    assert await _names(parallel_tool_calls=False) == ["get_weather"]
