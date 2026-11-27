import json
from typing import Any

import httpx2 as httpx
import openai
import pytest
from pydantic import ValidationError

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, inference
from livekit.agents.inference.llm import LLMStream, supports_prompt_cache_breakpoints
from livekit.agents.llm import AudioContent, CacheBreakpoint, ChatContext, ChatMessage, ImageContent
from livekit.agents.llm.chat_context import Instructions
from livekit.agents.voice.generation import update_instructions

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]

STATIC = "You are the Riverside Clinic voice agent. Follow the clinic rules."
DYNAMIC = "Current time: 09:01. Caller number: +15551234567."
IMAGE_URL = "https://example.com/insurance-card.jpg"
BREAKPOINT = {"mode": "explicit"}
IMAGE_PART = {"type": "image_url", "image_url": {"url": IMAGE_URL, "detail": "auto"}}
INPUT_IMAGE_PART = {"type": "input_image", "image_url": IMAGE_URL, "detail": "auto"}
IMAGE = ImageContent(image=IMAGE_URL)
MARK = CacheBreakpoint()


def _ctx(*content: object, role: str = "system") -> ChatContext:
    ctx = ChatContext()
    ctx.add_message(role=role, content=list(content))  # type: ignore[arg-type]
    return ctx


def _fmt(ctx: ChatContext, fmt: str = "openai", *, enabled: bool) -> list[dict]:
    messages, _ = ctx.to_provider_format(fmt, prompt_cache_breakpoints=enabled)
    return messages


def _tagged(text: str, text_type: str = "text") -> dict:
    return {"type": text_type, "text": text, "prompt_cache_breakpoint": BREAKPOINT}


def _plain(text: str, text_type: str = "text") -> dict:
    return {"type": text_type, "text": text}


def _sys(content: Any) -> list[dict]:
    return [{"role": "system", "content": content}]


@pytest.mark.parametrize(
    ("content", "enabled", "expected"),
    [
        ([STATIC, MARK, DYNAMIC], False, _sys(f"{STATIC}\n{DYNAMIC}")),
        ([STATIC, DYNAMIC], True, _sys(f"{STATIC}\n{DYNAMIC}")),
        ([STATIC, MARK, DYNAMIC], True, _sys([_tagged(STATIC), _plain(f"\n{DYNAMIC}")])),
        ([STATIC, MARK], True, _sys([_tagged(STATIC)])),
        ([MARK, STATIC], True, _sys(STATIC)),
        ([STATIC, MARK, MARK, DYNAMIC], True, _sys([_tagged(STATIC), _plain(f"\n{DYNAMIC}")])),
        (
            ["rules", MARK, "clinic hours", MARK, DYNAMIC],
            True,
            _sys([_tagged("rules"), _tagged("\nclinic hours"), _plain(f"\n{DYNAMIC}")]),
        ),
    ],
    ids=[
        "disabled-drops-marker",
        "no-marker-keeps-string",
        "tags-text-before-marker",
        "trailing-marker-tags-whole-message",
        "leading-marker-ignored",
        "repeated-markers-collapse",
        "multiple-segments",
    ],
)
def test_openai_format(content: list, enabled: bool, expected: list[dict]) -> None:
    assert _fmt(_ctx(*content), enabled=enabled) == expected


def test_openai_enabled_parts_join_back_to_unsplit_text() -> None:
    ctx = _ctx(STATIC, MARK, DYNAMIC)

    split = _fmt(ctx, enabled=True)[0]["content"]

    assert "".join(part["text"] for part in split) == _fmt(ctx, enabled=False)[0]["content"]


@pytest.mark.parametrize(
    ("content", "enabled", "expected"),
    [
        (
            [IMAGE, STATIC, MARK, DYNAMIC],
            True,
            [IMAGE_PART, _tagged(STATIC), _plain(f"\n{DYNAMIC}")],
        ),
        (
            [STATIC, MARK, IMAGE, DYNAMIC],
            True,
            [_tagged(STATIC), IMAGE_PART, _plain(f"\n{DYNAMIC}")],
        ),
        (
            [STATIC, IMAGE, MARK, DYNAMIC],
            True,
            [
                _plain(STATIC),
                {**IMAGE_PART, "prompt_cache_breakpoint": BREAKPOINT},
                _plain(f"\n{DYNAMIC}"),
            ],
        ),
        ([STATIC, MARK, IMAGE, DYNAMIC], False, [IMAGE_PART, _plain(f"{STATIC}\n{DYNAMIC}")]),
    ],
    ids=["image-before-prefix", "image-after-marker", "marker-after-image", "disabled"],
)
def test_openai_format_with_images(content: list, enabled: bool, expected: list[dict]) -> None:
    assert _fmt(_ctx(*content, role="user"), enabled=enabled)[0]["content"] == expected


def test_openai_enabled_tags_assistant_message_and_keeps_extra_content() -> None:
    ctx = ChatContext()
    ctx.add_message(
        role="assistant",
        content=["Hi, this is Rosie.", MARK],
        extra={"livekit": {"inference_deployment": "standard_openai"}},
    )

    assert _fmt(ctx, enabled=True) == [
        {
            "role": "assistant",
            "content": [_tagged("Hi, this is Rosie.")],
            "extra_content": {"livekit": {"inference_deployment": "standard_openai"}},
        }
    ]


@pytest.mark.parametrize(
    ("content", "role", "enabled", "expected"),
    [
        (
            [STATIC, MARK, DYNAMIC],
            "system",
            True,
            [_tagged(STATIC, "input_text"), _plain(f"\n{DYNAMIC}", "input_text")],
        ),
        (["Hi, this is Rosie.", MARK], "assistant", True, "Hi, this is Rosie."),
        ([STATIC, MARK, DYNAMIC], "system", False, f"{STATIC}\n{DYNAMIC}"),
    ],
    ids=["tags-input-text", "ignores-marker-on-assistant", "disabled-drops-marker"],
)
def test_responses_format(content: list, role: str, enabled: bool, expected: Any) -> None:
    items = _fmt(_ctx(*content, role=role), "openai.responses", enabled=enabled)

    assert items == [{"role": role, "content": expected}]


@pytest.mark.parametrize("fmt", ["google", "anthropic", "aws", "mistralai"])
def test_other_provider_formats_drop_marker(fmt: str) -> None:
    def conversation(*extra: CacheBreakpoint) -> ChatContext:
        ctx = ChatContext()
        ctx.add_message(role="system", content=[STATIC, *extra, DYNAMIC])
        ctx.add_message(role="user", content=["Hi, I need to reschedule.", *extra])
        return ctx

    assert conversation(MARK).to_provider_format(fmt) == conversation().to_provider_format(fmt)


def test_marker_round_trips_through_dict() -> None:
    data = _ctx(STATIC, MARK, DYNAMIC).to_dict()
    restored = ChatContext.from_dict(data)

    assert data["items"][0]["content"] == [STATIC, {"type": "cache_breakpoint"}, DYNAMIC]
    content = restored.items[0].content  # type: ignore[union-attr]
    assert content[0] == STATIC and content[2] == DYNAMIC
    assert isinstance(content[1], CacheBreakpoint)


def test_text_content_ignores_marker() -> None:
    message = _ctx(STATIC, MARK, DYNAMIC).items[0]

    assert message.text_content == f"{STATIC}\n{DYNAMIC}"  # type: ignore[union-attr]


@pytest.mark.parametrize("content", [{"text": "hi"}, {"type": "cache_breakpoint", "text": "hi"}])
def test_malformed_dict_content_is_rejected(content: dict) -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content=[content])  # type: ignore[list-item]
    with pytest.raises(ValidationError):
        ChatContext.from_dict(
            {"items": [{"type": "message", "role": "user", "content": ["hello", content]}]}
        )


# marker-free output must not drift: every request goes through the breakpoint-aware formatter
_AUDIO = AudioContent(
    frame=[
        rtc.AudioFrame(data=b"\x00\x00", sample_rate=16000, num_channels=1, samples_per_channel=1)
    ]
)


@pytest.mark.parametrize(
    ("content", "chat_content", "responses_content"),
    [
        ([IMAGE], [IMAGE_PART], [INPUT_IMAGE_PART]),
        (["a", IMAGE], [IMAGE_PART, _plain("a")], [INPUT_IMAGE_PART, _plain("a", "input_text")]),
        (
            [IMAGE, "a", IMAGE],
            [IMAGE_PART, IMAGE_PART, _plain("a")],
            [INPUT_IMAGE_PART, INPUT_IMAGE_PART, _plain("a", "input_text")],
        ),
        (["a", _AUDIO, "b"], "a\nb", "a\nb"),
        (["", "a"], "a", "a"),
        ([], "", ""),
    ],
    ids=[
        "image-only",
        "text-then-image",
        "images-around-text",
        "audio-skipped",
        "leading-empty-string",
        "empty",
    ],
)
def test_marker_free_output(content: list, chat_content: Any, responses_content: Any) -> None:
    ctx = _ctx(*content, role="user")
    for fmt, expected in (("openai", chat_content), ("openai.responses", responses_content)):
        for enabled in (False, True):
            assert _fmt(ctx, fmt, enabled=enabled) == [{"role": "user", "content": expected}]


# ---------------------------------------------------------------------------
# inference LLM: which models get breakpoints on the wire
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "supported"),
    [
        ("openai/gpt-5.6-luna", True),
        ("gpt-5.6", True),
        ("gpt-5.7", True),
        ("gpt-6", True),
        ("openai/gpt-6-luna", True),
        ("gpt-7-mini", True),
        ("gpt-5.10", True),
        ("OpenAI/GPT-5.6-Luna", True),
        ("gpt-5.6-chat-latest", True),
        ("openai/some-new-model", False),
        ("openai/gpt-4.1", False),
        ("gpt-4.1", False),
        ("gpt-5", False),
        ("gpt-5.5-pro", False),
        ("gpt-5.1-chat-latest", False),
        ("openai/chat-latest", False),
        ("chatgpt-4o-latest", False),
        ("gpt-4o-mini", False),
        ("o3", False),
        ("openai/gpt-oss-120b", False),
        ("azure/gpt-5.6-luna", False),
        ("google/gemini-3.5-flash", False),
        ("deepseek-ai/deepseek-v3.2", False),
    ],
)
def test_supports_prompt_cache_breakpoints(model: str, supported: bool) -> None:
    assert supports_prompt_cache_breakpoints(model) is supported


_SSE = (
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n\n'
    "data: [DONE]\n\n"
)


def _llm(model: str, **kwargs: Any) -> tuple[inference.LLM, dict[str, Any]]:
    llm = inference.LLM(
        model, base_url="http://gateway.test/v1", api_key="key", api_secret="s" * 32, **kwargs
    )
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_SSE)

    llm._client = openai.AsyncClient(
        api_key="token",
        base_url="http://gateway.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return llm, captured


def _wire_ctx() -> ChatContext:
    # the marker ends the static system message; per-call text is its own message
    ctx = ChatContext()
    ctx.add_message(role="system", content=[STATIC, MARK])
    ctx.add_message(role="system", content=DYNAMIC)
    ctx.add_message(role="user", content="Hi, I need to reschedule.")
    return ctx


async def _run(llm: inference.LLM, stream: LLMStream) -> None:
    async with stream:
        async for _ in stream:
            pass
    await llm.aclose()


@pytest.mark.parametrize(
    ("model", "kwargs", "update", "sent"),
    [
        ("openai/gpt-5.6-luna", {}, None, True),
        ("openai/gpt-4.1", {}, None, False),
        ("google/gemini-3.5-flash", {}, None, False),
        ("openai/gpt-5.6-luna", {"prompt_cache_breakpoints": False}, None, False),
        ("openai/gpt-4.1", {"prompt_cache_breakpoints": True}, None, True),
        ("openai/gpt-5.6-luna", {}, {"model": "openai/gpt-4.1"}, False),
        ("openai/gpt-5.6-luna", {}, {"prompt_cache_breakpoints": False}, False),
    ],
)
async def test_wire_breakpoints(model: str, kwargs: dict, update: dict | None, sent: bool) -> None:
    llm, body = _llm(model, **kwargs)
    if update:
        llm.update_options(**update)
    await _run(llm, llm.chat(chat_ctx=_wire_ctx()))

    assert body["messages"][0]["content"] == ([_tagged(STATIC)] if sent else STATIC)
    assert body["messages"][1]["content"] == DYNAMIC


async def test_stream_ignores_flag_for_non_openai_format() -> None:
    llm, body = _llm("openai/gpt-5.6-luna")
    stream = LLMStream(
        llm,
        model="google/gemini-3.5-flash",
        strict_tool_schema=False,
        client=llm._client,
        chat_ctx=_wire_ctx(),
        tools=[],
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        extra_kwargs={},
        provider_fmt="google",
        prompt_cache_breakpoints=True,
    )
    await _run(llm, stream)

    assert "prompt_cache_breakpoint" not in json.dumps(body)


async def test_wire_folds_dynamic_instructions_for_google_model() -> None:
    # the gateway keeps one system prompt for Gemini and Gemma; a second one becomes a user turn
    llm, body = _llm("google/gemini-3.5-flash")
    ctx = ChatContext()
    update_instructions(
        ctx, instructions=Instructions(STATIC, dynamic=DYNAMIC), add_if_missing=True
    )
    ctx.add_message(role="user", content="Hi, I need to reschedule.")
    await _run(llm, llm.chat(chat_ctx=ctx))

    assert body["messages"][0] == {"role": "system", "content": f"{STATIC}\n{DYNAMIC}"}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
