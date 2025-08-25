from __future__ import annotations

import asyncio
import base64
from enum import Enum
from pathlib import Path
from typing import Annotated, Callable
from dataclasses import dataclass, field

import pytest

from livekit.agents import APIConnectionError, llm
from livekit.agents.llm import ChatContext, FunctionTool, function_tool, ImageContent
from livekit.plugins import anthropic, aws, google, mistralai, openai
from livekit.rtc import VideoBufferType, VideoFrame


class Unit(Enum):
    FAHRENHEIT = "fahrenheit"
    CELSIUS = "celsius"


@dataclass(frozen=True)
class TypeInfo:
    description: str = ""
    choices: list[Any] = field(default_factory=list, hash=False)


class FncTls():
    @function_tool(name="get_weather", description="Get the current weather in a given location")
    def get_weather(
        location: Annotated[
            str, TypeInfo(description="The city and state, e.g. San Francisco, CA")
        ],
        unit: Annotated[Unit, TypeInfo(description="The temperature unit to use.")] = Unit.CELSIUS,
    ) -> None: ...

    @function_tool(name="play_music", description="Play a music")
    def play_music(
        name: Annotated[str, TypeInfo(description="the name of the Artist")],
    ) -> None: ...

    # test for cancelled calls
    @function_tool(name="toggle_light", description="Turn on/off the lights in a room")
    async def toggle_light(
        room: Annotated[str, TypeInfo(description="The room to control")],
        on: bool = True,
    ) -> FunctionTool:
        await asyncio.sleep(60)

    # used to test arrays as arguments
    @function_tool(name="select_currencies", description="Select currencies of a specific area")
    def select_currencies(
        currencies: Annotated[
            list[str],
            TypeInfo(
                description="The currencies to select",
                choices=["usd", "eur", "gbp", "jpy", "sek"],
            ),
        ],
    ) -> None: ...

    @function_tool(name="update_user_info", description="Update user info")
    def update_user_info(
        email: Annotated[str | None, TypeInfo(description="The user address email")] = None,
        name: Annotated[str | None, TypeInfo(description="The user name")] = None,
        address: Annotated[str, TypeInfo(description="The user address")] | None = None,
    ) -> None: ...


def test_hashable_typeinfo():
    typeinfo = TypeInfo(description="testing", choices=[1, 2, 3])
    # TypeInfo must be hashable when used in combination of typing.Annotated
    hash(typeinfo)


LLMS: list[Callable[[], llm.LLM]] = [
    pytest.param(lambda: openai.LLM(), id="openai"),
    # lambda: openai.beta.AssistantLLM(
    #     assistant_opts=openai.beta.AssistantOptions(
    #         create_options=openai.beta.AssistantCreateOptions(
    #             name=f"test-{uuid.uuid4()}",
    #             instructions="You are a basic assistant",
    #             model="gpt-4o",
    #         )
    #     )
    # ),
    pytest.param(lambda: anthropic.LLM(), id="anthropic"),
    pytest.param(lambda: google.LLM(), id="google"),
    pytest.param(lambda: google.LLM(vertexai=True), id="google-vertexai"),
    # pytest.param(lambda: aws.LLM(), id="aws"),
    # pytest.param(lambda: mistralai.LLM(), id="mistralai"),
]


@pytest.mark.parametrize("llm_factory", LLMS)
async def test_chat(llm_factory: Callable[[], llm.LLM]):
    input_llm = llm_factory()
    chat_ctx = ChatContext()
    chat_ctx.add_message(
        content='You are an assistant at a drive-thru restaurant "Live-Burger". Ask the customer what they would like to order.',  # noqa: E501
        role="assistant",
    )

    # Anthropic and vertex requires at least one message (system messages don't count)
    chat_ctx.add_message(
        content="Hello",
        role="user",
    )

    stream = input_llm.chat(chat_ctx=chat_ctx)
    text = ""
    async for chunk in stream:
        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content
        if content:
            text += content

    assert len(text) > 0


@pytest.mark.parametrize("llm_factory", LLMS)
async def test_llm_chat_with_consecutive_messages(
    llm_factory: callable,
):
    input_llm = llm_factory()

    chat_ctx = ChatContext()
    chat_ctx.add_message(
        content="Hello, How can I help you today?",
        role="assistant",
    )
    chat_ctx.add_message(content="I see that you have a busy day ahead.", role="assistant")
    chat_ctx.add_message(content="Actually, I need some help with my recent order.", role="user")
    chat_ctx.add_message(content="I want to cancel my order.", role="user")

    stream = input_llm.chat(chat_ctx=chat_ctx)
    collected_text = ""
    async for chunk in stream:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            collected_text += content

    assert len(collected_text) > 0, "Expected a non-empty response from the LLM chat"


@pytest.mark.parametrize("llm_factory", LLMS)
async def test_basic_fnc_calls(llm_factory: Callable[[], llm.LLM]):
    input_llm = llm_factory()
    fnctls = [FncTls.get_weather]

    stream = await _request_fnc_call(
        input_llm,
        "What's the weather in San Francisco and what's the weather Paris?",
        fnctls,
    )
    calls = stream.execute_functions()
    await asyncio.gather(*[f.task for f in calls])
    await stream.aclose()
    assert len(calls) == 2, "get_weather should be called twice"


@pytest.mark.parametrize("llm_factory", LLMS)
async def test_function_call_exception_handling(llm_factory: Callable[[], llm.LLM]):
    input_llm = llm_factory()

    @function_tool(description="Simulate a failure")
    async def failing_function():
        raise RuntimeError("Simulated failure")

    stream = await _request_fnc_call(input_llm, "Call the failing function", failing_function)
    calls = stream.execute_functions()
    await asyncio.gather(*[f.task for f in calls], return_exceptions=True)
    await stream.aclose()

    assert len(calls) == 1
    assert isinstance(calls[0].exception, RuntimeError)
    assert str(calls[0].exception) == "Simulated failure"


@pytest.mark.parametrize("llm_factory", LLMS)
async def test_runtime_addition(llm_factory: Callable[[], llm.LLM]):
    input_llm = llm_factory()
    called_msg = ""

    @function_tool(description="Show a message on the screen")
    async def show_message(
        message: Annotated[str, TypeInfo(description="The message to show")],
    ):
        nonlocal called_msg
        called_msg = message

    stream = await _request_fnc_call(
        input_llm, "Can you show 'Hello LiveKit!' on the screen?", show_message
    )
    fns = stream.execute_functions()
    await asyncio.gather(*[f.task for f in fns])
    await stream.aclose()

    assert called_msg == "Hello LiveKit!", "send_message should be called"


@pytest.mark.parametrize("llm_factory", LLMS)
async def test_cancelled_calls(llm_factory: Callable[[], llm.LLM]):
    input_llm = llm_factory()
    fnctls = [FncTls.toggle_light]

    stream = await _request_fnc_call(input_llm, "Turn off the lights in the bedroom", fnctls)
    calls = stream.execute_functions()
    await asyncio.sleep(0.2)  # wait for the loop executor to start the task

    # don't wait for gather_function_results and directly close (this should cancel the
    # ongoing calls)
    await stream.aclose()

    assert len(calls) == 1
    assert isinstance(calls[0].exception, asyncio.CancelledError), (
        "toggle_light should have been cancelled"
    )


@pytest.mark.parametrize("llm_factory", LLMS)
async def test_calls_arrays(llm_factory: Callable[[], llm.LLM]):
    input_llm = llm_factory()
    fnctls = [FncTls.select_currencies]

    stream = await _request_fnc_call(
        input_llm,
        "Can you select all currencies in Europe at once from given choices using function call `select_currencies`?",  # noqa: E501
        fnctls,
        temperature=0.2,
    )
    calls = stream.execute_functions()
    await asyncio.gather(*[f.task for f in calls])
    await stream.aclose()

    assert len(calls) == 1, "select_currencies should have been called only once"

    call = calls[0]
    currencies = call.call_info.arguments["currencies"]
    assert len(currencies) == 3, "select_currencies should have 3 currencies"
    assert "eur" in currencies and "gbp" in currencies and "sek" in currencies, (
        "select_currencies should have eur, gbp, sek"
    )


@pytest.mark.parametrize("llm_factory", LLMS)
async def test_calls_choices(llm_factory: Callable[[], llm.LLM]):
    input_llm = llm_factory()

    # test choices on int
    @function_tool(description="Change the volume")
    def change_volume(
        volume: Annotated[
            int, TypeInfo(description="The volume level", choices=[0, 11, 30, 83, 99])
        ],
    ) -> None: ...

    if not input_llm.capabilities.supports_choices_on_int:
        with pytest.raises(APIConnectionError):
            stream = await _request_fnc_call(input_llm, "Set the volume to 30", change_volume)
    else:
        stream = await _request_fnc_call(input_llm, "Set the volume to 30", change_volume)
        calls = stream.execute_functions()
        await asyncio.gather(*[f.task for f in calls])
        await stream.aclose()

        assert len(calls) == 1, "change_volume should have been called only once"

        call = calls[0]
        volume = call.call_info.arguments["volume"]
        assert volume == 30, "change_volume should have been called with volume 30"


@pytest.mark.parametrize("llm_factory", LLMS)
async def test_optional_args(llm_factory: Callable[[], llm.LLM]):
    input_llm = llm_factory()
    fnctls = [FncTls.update_user_info]

    stream = await _request_fnc_call(
        input_llm, "Using a tool call update the user info to name Theo", fnctls
    )

    calls = stream.execute_functions()
    await asyncio.gather(*[f.task for f in calls])
    await stream.aclose()

    assert len(calls) == 1, "update_user_info should have been called only once"

    call = calls[0]
    name = call.call_info.arguments.get("name", None)
    email = call.call_info.arguments.get("email", None)
    address = call.call_info.arguments.get("address", None)

    assert name == "Theo", "update_user_info should have been called with name 'Theo'"
    assert email is None, "update_user_info should have been called with email None"
    assert address is None, "update_user_info should have been called with address None"


test_tool_choice_cases = [
    pytest.param(
        "Default tool_choice (auto)",
        "Get the weather for New York and play some music from the artist 'The Beatles'.",
        None,
        {"get_weather", "play_music"},
        id="Default tool_choice (auto)",
    ),
    pytest.param(
        "Tool_choice set to 'required'",
        "Get the weather for Chicago and play some music from the artist 'Eminem'.",
        "required",
        {"get_weather", "play_music"},
        id="Tool_choice set to 'required'",
    ),
    pytest.param(
        "Tool_choice set to a specific tool ('get_weather')",
        "Get the weather for Miami.",
        llm.tool_context.NamedToolChoice(type="function", function="get_weather"),
        {"get_weather"},
        id="Tool_choice set to a specific tool ('get_weather')",
    ),
    pytest.param(
        "Tool_choice set to 'none'",
        "Get the weather for Seattle and play some music from the artist 'Frank Sinatra'.",
        "none",
        set(),  # No tool calls expected
        id="Tool_choice set to 'none'",
    ),
]


@pytest.mark.parametrize(
    "description, user_request, tool_choice, expected_calls", test_tool_choice_cases
)
@pytest.mark.parametrize("llm_factory", LLMS)
async def test_tool_choice_options(
    description: str,
    user_request: str,
    tool_choice: dict | str | None,
    expected_calls: set,
    llm_factory: Callable[[], llm.LLM],
):
    input_llm = llm_factory()
    fnctls = [FncTls.get_weather, FncTls.play_music]

    stream = await _request_fnc_call(
        input_llm,
        user_request,
        fnctls,
        tool_choice=tool_choice,
        parallel_tool_calls=True,
    )

    calls = stream.execute_functions()
    await asyncio.gather(*[f.task for f in calls], return_exceptions=True)
    await stream.aclose()
    print(calls)

    call_names = {call.call_info.function_info.name for call in calls}
    if tool_choice == "none":
        assert call_names == expected_calls, (
            f"Test '{description}' failed: Expected calls {expected_calls}, but got {call_names}"
        )


async def _request_fnc_call(
    model: llm.LLM,
    request: str,
    tools: FunctionTool,
    temperature: float | None = None,
    parallel_tool_calls: bool | None = None,
    tool_choice: llm.NamedToolChoice | None = None,
) -> llm.LLMStream:
    chat_ctx=ChatContext()
    chat_ctx.add_message(
        content="You are an helpful assistant. Follow the instructions provided by the user. You can use multiple tool calls at once.",  # noqa: E501
        role="system",
    )
    chat_ctx.add_message(content=request, role="user")
    stream = model.chat(
        chat_ctx=chat_ctx,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        extra_kwargs={"temperature": temperature},
    )

    async for _ in stream:
        pass

    return stream


_HEARTS_RGBA_PATH = Path(__file__).parent / "hearts.rgba"
with open(_HEARTS_RGBA_PATH, "rb") as f:
    image_data = f.read()

    _HEARTS_IMAGE_VIDEO_FRAME = VideoFrame(
        width=512, height=512, type=VideoBufferType.RGBA, data=image_data
    )

_HEARTS_JPEG_PATH = Path(__file__).parent / "hearts.jpg"
with open(_HEARTS_JPEG_PATH, "rb") as f:
    _HEARTS_IMAGE_DATA_URL = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"


@pytest.mark.parametrize("llm_factory", LLMS)
async def test_chat_with_image_data_url(llm_factory: Callable[[], llm.LLM]):
    input_llm = llm_factory()

    chat_ctx = ChatContext()
    chat_ctx.add_message(
        content="You are an AI assistant that describes images in detail upon request.",
        role="system",
    )
    chat_ctx.add_message(
        content=["Describe this image", ImageContent(image=_HEARTS_IMAGE_DATA_URL, inference_detail="low")],
        role="user",
    )

    stream = input_llm.chat(chat_ctx=chat_ctx)
    text = ""
    async for chunk in stream:
        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content
        if content:
            text += content

    assert "heart" in text.lower()


@pytest.mark.parametrize("llm_factory", LLMS)
async def test_chat_with_image_frame(llm_factory: Callable[[], llm.LLM]):
    input_llm = llm_factory()

    chat_ctx = ChatContext()
    chat_ctx.add_message(
        content="You are an AI assistant that describes images in detail upon request.",
        role="system",
    )
    chat_ctx.add_message(
        content=["Describe this image", ImageContent(image=_HEARTS_IMAGE_VIDEO_FRAME, inference_detail="low")],
        role="user",
    )

    stream = input_llm.chat(chat_ctx=chat_ctx)
    text = ""
    async for chunk in stream:
        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content
        if content:
            text += content

    assert "heart" in text.lower()
