import pytest
from pydantic import ValidationError

from livekit import rtc
from livekit.agents.llm import AudioContent, CacheBreakpoint, ChatContext, ChatMessage, ImageContent

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]

STATIC = "You are the Riverside Clinic voice agent. Follow the clinic rules."
DYNAMIC = "Current time: 09:01. Caller number: +15551234567."
IMAGE_URL = "https://example.com/insurance-card.jpg"
BREAKPOINT = {"mode": "explicit"}
IMAGE_PART = {"type": "image_url", "image_url": {"url": IMAGE_URL, "detail": "auto"}}
INPUT_IMAGE_PART = {"type": "input_image", "image_url": IMAGE_URL, "detail": "auto"}


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


def _image_part(part: dict) -> dict:
    assert part["type"] == "image_url"
    assert part["image_url"]["url"] == IMAGE_URL
    return part


def test_openai_enabled_image_before_prefix_stays_first():
    ctx = _ctx(ImageContent(image=IMAGE_URL), STATIC, CacheBreakpoint(), DYNAMIC, role="user")

    content = _openai(ctx, enabled=True)[0]["content"]

    _image_part(content[0])
    assert "prompt_cache_breakpoint" not in content[0]
    assert content[1:] == [_tagged(STATIC), _plain(f"\n{DYNAMIC}")]


def test_openai_enabled_image_after_marker_stays_out_of_prefix():
    ctx = _ctx(STATIC, CacheBreakpoint(), ImageContent(image=IMAGE_URL), DYNAMIC, role="user")

    content = _openai(ctx, enabled=True)[0]["content"]

    assert content[0] == _tagged(STATIC)
    _image_part(content[1])
    assert "prompt_cache_breakpoint" not in content[1]
    assert content[2] == _plain(f"\n{DYNAMIC}")


def test_openai_enabled_marker_after_image_tags_image():
    ctx = _ctx(STATIC, ImageContent(image=IMAGE_URL), CacheBreakpoint(), DYNAMIC, role="user")

    content = _openai(ctx, enabled=True)[0]["content"]

    assert content[0] == _plain(STATIC)
    assert _image_part(content[1])["prompt_cache_breakpoint"] == BREAKPOINT
    assert content[2] == _plain(f"\n{DYNAMIC}")


def test_openai_disabled_with_image_keeps_images_first():
    ctx = _ctx(STATIC, CacheBreakpoint(), ImageContent(image=IMAGE_URL), DYNAMIC, role="user")

    content = _openai(ctx, enabled=False)[0]["content"]

    _image_part(content[0])
    assert content[1:] == [_plain(f"{STATIC}\n{DYNAMIC}")]


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


def test_typeless_dict_content_is_rejected_by_chat_message():
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content=[{"text": "hi"}])  # type: ignore[list-item]


def test_typeless_dict_content_is_rejected_by_from_dict():
    data = {"items": [{"type": "message", "role": "user", "content": ["hello", {"text": "world"}]}]}

    with pytest.raises(ValidationError):
        ChatContext.from_dict(data)


def test_marker_dict_with_extra_field_is_rejected():
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content=[{"type": "cache_breakpoint", "text": "hi"}])  # type: ignore[list-item]


# The literal outputs below are what main produced before CacheBreakpoint existed. Every
# request goes through the rewritten formatter, so the marker-free path must not drift.


def _audio() -> AudioContent:
    frame = rtc.AudioFrame(
        data=b"\x00\x00", sample_rate=16000, num_channels=1, samples_per_channel=1
    )
    return AudioContent(frame=[frame])


def _default_openai(ctx: ChatContext) -> list[dict]:
    disabled = _openai(ctx, enabled=False)
    assert _openai(ctx, enabled=True) == disabled
    return disabled


def _default_responses(ctx: ChatContext) -> list[dict]:
    disabled = _responses(ctx, enabled=False)
    assert _responses(ctx, enabled=True) == disabled
    return disabled


def test_default_openai_image_only():
    ctx = _ctx(ImageContent(image=IMAGE_URL), role="user")

    assert _default_openai(ctx) == [{"role": "user", "content": [IMAGE_PART]}]


def test_default_responses_image_only():
    ctx = _ctx(ImageContent(image=IMAGE_URL), role="user")

    assert _default_responses(ctx) == [{"role": "user", "content": [INPUT_IMAGE_PART]}]


def test_default_openai_text_then_image_puts_image_first():
    ctx = _ctx("a", ImageContent(image=IMAGE_URL), role="user")

    assert _default_openai(ctx) == [
        {"role": "user", "content": [IMAGE_PART, {"type": "text", "text": "a"}]}
    ]


def test_default_responses_text_then_image_puts_image_first():
    ctx = _ctx("a", ImageContent(image=IMAGE_URL), role="user")

    assert _default_responses(ctx) == [
        {"role": "user", "content": [INPUT_IMAGE_PART, {"type": "input_text", "text": "a"}]}
    ]


def test_default_openai_images_around_text_group_images_first():
    ctx = _ctx(ImageContent(image=IMAGE_URL), "a", ImageContent(image=IMAGE_URL), role="user")

    assert _default_openai(ctx) == [
        {"role": "user", "content": [IMAGE_PART, IMAGE_PART, {"type": "text", "text": "a"}]}
    ]


def test_default_responses_images_around_text_group_images_first():
    ctx = _ctx(ImageContent(image=IMAGE_URL), "a", ImageContent(image=IMAGE_URL), role="user")

    assert _default_responses(ctx) == [
        {
            "role": "user",
            "content": [INPUT_IMAGE_PART, INPUT_IMAGE_PART, {"type": "input_text", "text": "a"}],
        }
    ]


def test_default_openai_audio_is_skipped_and_text_joined_with_newline():
    ctx = _ctx("a", _audio(), "b", role="user")

    assert _default_openai(ctx) == [{"role": "user", "content": "a\nb"}]


def test_default_responses_audio_is_skipped_and_text_joined_with_newline():
    ctx = _ctx("a", _audio(), "b", role="user")

    assert _default_responses(ctx) == [{"role": "user", "content": "a\nb"}]


def test_default_openai_leading_empty_string_adds_no_newline():
    ctx = _ctx("", "a", role="user")

    assert _default_openai(ctx) == [{"role": "user", "content": "a"}]


def test_default_responses_leading_empty_string_adds_no_newline():
    ctx = _ctx("", "a", role="user")

    assert _default_responses(ctx) == [{"role": "user", "content": "a"}]


def test_default_openai_empty_content_is_empty_string():
    ctx = _ctx(role="user")

    assert _default_openai(ctx) == [{"role": "user", "content": ""}]


def test_default_responses_empty_content_is_empty_string():
    ctx = _ctx(role="user")

    assert _default_responses(ctx) == [{"role": "user", "content": ""}]
