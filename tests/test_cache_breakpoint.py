import pytest

from livekit.agents.llm import CacheBreakpoint, ChatContext, ImageContent

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]

STATIC = "You are the Riverside Clinic voice agent. Follow the clinic rules."
DYNAMIC = "Current time: 09:01. Caller number: +15551234567."
IMAGE_URL = "https://example.com/insurance-card.jpg"
BREAKPOINT = {"mode": "explicit"}


def _ctx(*content: object, role: str = "system") -> ChatContext:
    ctx = ChatContext()
    ctx.add_message(role=role, content=list(content))  # type: ignore[arg-type]
    return ctx


def _openai(ctx: ChatContext, *, enabled: bool) -> list[dict]:
    messages, _ = ctx.to_provider_format("openai", prompt_cache_breakpoints=enabled)
    return messages


def _responses(ctx: ChatContext, *, enabled: bool) -> list[dict]:
    items, _ = ctx.to_provider_format("openai.responses", prompt_cache_breakpoints=enabled)
    return items


def _tagged(text: str, text_type: str = "text") -> dict:
    return {"type": text_type, "text": text, "prompt_cache_breakpoint": BREAKPOINT}


def _plain(text: str, text_type: str = "text") -> dict:
    return {"type": text_type, "text": text}


def test_openai_disabled_drops_marker():
    messages = _openai(_ctx(STATIC, CacheBreakpoint(), DYNAMIC), enabled=False)

    assert messages == [{"role": "system", "content": f"{STATIC}\n{DYNAMIC}"}]


def test_openai_disabled_matches_context_without_marker():
    with_marker = _openai(_ctx(STATIC, CacheBreakpoint(), DYNAMIC), enabled=False)
    without_marker = _openai(_ctx(STATIC, DYNAMIC), enabled=False)

    assert with_marker == without_marker


def test_openai_enabled_without_marker_keeps_string_content():
    messages = _openai(_ctx(STATIC, DYNAMIC), enabled=True)

    assert messages == [{"role": "system", "content": f"{STATIC}\n{DYNAMIC}"}]


def test_openai_enabled_tags_text_before_marker():
    messages = _openai(_ctx(STATIC, CacheBreakpoint(), DYNAMIC), enabled=True)

    assert messages == [{"role": "system", "content": [_tagged(STATIC), _plain(f"\n{DYNAMIC}")]}]


def test_openai_enabled_parts_join_back_to_unsplit_text():
    ctx = _ctx(STATIC, CacheBreakpoint(), DYNAMIC)

    split = _openai(ctx, enabled=True)[0]["content"]
    unsplit = _openai(ctx, enabled=False)[0]["content"]

    assert "".join(part["text"] for part in split) == unsplit


def test_openai_enabled_trailing_marker_tags_whole_message():
    messages = _openai(_ctx(STATIC, CacheBreakpoint()), enabled=True)

    assert messages == [{"role": "system", "content": [_tagged(STATIC)]}]


def test_openai_enabled_leading_marker_is_ignored():
    messages = _openai(_ctx(CacheBreakpoint(), STATIC), enabled=True)

    assert messages == [{"role": "system", "content": STATIC}]


def test_openai_enabled_repeated_markers_collapse():
    messages = _openai(_ctx(STATIC, CacheBreakpoint(), CacheBreakpoint(), DYNAMIC), enabled=True)

    assert messages == [{"role": "system", "content": [_tagged(STATIC), _plain(f"\n{DYNAMIC}")]}]


def test_openai_enabled_multiple_markers_make_multiple_segments():
    ctx = _ctx("rules", CacheBreakpoint(), "clinic hours", CacheBreakpoint(), DYNAMIC)

    messages = _openai(ctx, enabled=True)

    assert messages == [
        {
            "role": "system",
            "content": [_tagged("rules"), _tagged("\nclinic hours"), _plain(f"\n{DYNAMIC}")],
        }
    ]


def test_openai_enabled_keeps_images_before_text_parts():
    ctx = _ctx(ImageContent(image=IMAGE_URL), STATIC, CacheBreakpoint(), DYNAMIC, role="user")

    content = _openai(ctx, enabled=True)[0]["content"]

    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"] == IMAGE_URL
    assert content[1:] == [_tagged(STATIC), _plain(f"\n{DYNAMIC}")]


def test_openai_enabled_tags_assistant_message():
    messages = _openai(
        _ctx("Hi, this is Rosie.", CacheBreakpoint(), role="assistant"), enabled=True
    )

    assert messages == [{"role": "assistant", "content": [_tagged("Hi, this is Rosie.")]}]


def test_openai_enabled_keeps_extra_content():
    ctx = ChatContext()
    ctx.add_message(
        role="assistant",
        content=["Hi, this is Rosie.", CacheBreakpoint()],
        extra={"livekit": {"inference_deployment": "standard_openai"}},
    )

    message = _openai(ctx, enabled=True)[0]

    assert message["extra_content"] == {"livekit": {"inference_deployment": "standard_openai"}}


def test_responses_enabled_tags_input_text_before_marker():
    items = _responses(_ctx(STATIC, CacheBreakpoint(), DYNAMIC), enabled=True)

    assert items == [
        {
            "role": "system",
            "content": [_tagged(STATIC, "input_text"), _plain(f"\n{DYNAMIC}", "input_text")],
        }
    ]


def test_responses_enabled_ignores_marker_on_assistant():
    items = _responses(
        _ctx("Hi, this is Rosie.", CacheBreakpoint(), role="assistant"), enabled=True
    )

    assert items == [{"role": "assistant", "content": "Hi, this is Rosie."}]


def test_responses_disabled_drops_marker():
    items = _responses(_ctx(STATIC, CacheBreakpoint(), DYNAMIC), enabled=False)

    assert items == [{"role": "system", "content": f"{STATIC}\n{DYNAMIC}"}]


def _conversation(*, marker: bool) -> ChatContext:
    extra = [CacheBreakpoint()] if marker else []
    ctx = ChatContext()
    ctx.add_message(role="system", content=[STATIC, *extra, DYNAMIC])
    ctx.add_message(role="user", content=["Hi, I need to reschedule.", *extra])
    return ctx


def test_google_format_drops_marker():
    assert _conversation(marker=True).to_provider_format("google") == _conversation(
        marker=False
    ).to_provider_format("google")


def test_anthropic_format_drops_marker():
    assert _conversation(marker=True).to_provider_format("anthropic") == _conversation(
        marker=False
    ).to_provider_format("anthropic")


def test_aws_format_drops_marker():
    assert _conversation(marker=True).to_provider_format("aws") == _conversation(
        marker=False
    ).to_provider_format("aws")


def test_mistralai_format_drops_marker():
    assert _conversation(marker=True).to_provider_format("mistralai") == _conversation(
        marker=False
    ).to_provider_format("mistralai")


def test_marker_round_trips_through_dict():
    restored = ChatContext.from_dict(_ctx(STATIC, CacheBreakpoint(), DYNAMIC).to_dict())

    content = restored.items[0].content  # type: ignore[union-attr]
    assert content[0] == STATIC
    assert isinstance(content[1], CacheBreakpoint)
    assert content[2] == DYNAMIC


def test_marker_serializes_with_its_type():
    data = _ctx(STATIC, CacheBreakpoint()).to_dict()

    assert data["items"][0]["content"] == [STATIC, {"type": "cache_breakpoint"}]


def test_text_content_ignores_marker():
    message = _ctx(STATIC, CacheBreakpoint(), DYNAMIC).items[0]

    assert message.text_content == f"{STATIC}\n{DYNAMIC}"  # type: ignore[union-attr]
