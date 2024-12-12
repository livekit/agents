from __future__ import annotations

import asyncio
from enum import Enum
from typing import Annotated, Callable, Literal, Optional, Union

import pytest
from livekit.agents import APIConnectionError, llm
from livekit.agents.llm import ChatContext, FunctionContext, TypeInfo, ai_callable
from livekit.plugins import anthropic, openai

SAMPLE_IMAGE_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAApgAAAKYB3X3/OAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAANCSURBVEiJtZZPbBtFFMZ/M7ubXdtdb1xSFyeilBapySVU8h8OoFaooFSqiihIVIpQBKci6KEg9Q6H9kovIHoCIVQJJCKE1ENFjnAgcaSGC6rEnxBwA04Tx43t2FnvDAfjkNibxgHxnWb2e/u992bee7tCa00YFsffekFY+nUzFtjW0LrvjRXrCDIAaPLlW0nHL0SsZtVoaF98mLrx3pdhOqLtYPHChahZcYYO7KvPFxvRl5XPp1sN3adWiD1ZAqD6XYK1b/dvE5IWryTt2udLFedwc1+9kLp+vbbpoDh+6TklxBeAi9TL0taeWpdmZzQDry0AcO+jQ12RyohqqoYoo8RDwJrU+qXkjWtfi8Xxt58BdQuwQs9qC/afLwCw8tnQbqYAPsgxE1S6F3EAIXux2oQFKm0ihMsOF71dHYx+f3NND68ghCu1YIoePPQN1pGRABkJ6Bus96CutRZMydTl+TvuiRW1m3n0eDl0vRPcEysqdXn+jsQPsrHMquGeXEaY4Yk4wxWcY5V/9scqOMOVUFthatyTy8QyqwZ+kDURKoMWxNKr2EeqVKcTNOajqKoBgOE28U4tdQl5p5bwCw7BWquaZSzAPlwjlithJtp3pTImSqQRrb2Z8PHGigD4RZuNX6JYj6wj7O4TFLbCO/Mn/m8R+h6rYSUb3ekokRY6f/YukArN979jcW+V/S8g0eT/N3VN3kTqWbQ428m9/8k0P/1aIhF36PccEl6EhOcAUCrXKZXXWS3XKd2vc/TRBG9O5ELC17MmWubD2nKhUKZa26Ba2+D3P+4/MNCFwg59oWVeYhkzgN/JDR8deKBoD7Y+ljEjGZ0sosXVTvbc6RHirr2reNy1OXd6pJsQ+gqjk8VWFYmHrwBzW/n+uMPFiRwHB2I7ih8ciHFxIkd/3Omk5tCDV1t+2nNu5sxxpDFNx+huNhVT3/zMDz8usXC3ddaHBj1GHj/As08fwTS7Kt1HBTmyN29vdwAw+/wbwLVOJ3uAD1wi/dUH7Qei66PfyuRj4Ik9is+hglfbkbfR3cnZm7chlUWLdwmprtCohX4HUtlOcQjLYCu+fzGJH2QRKvP3UNz8bWk1qMxjGTOMThZ3kvgLI5AzFfo379UAAAAASUVORK5CYII="

class Unit(Enum):
    FAHRENHEIT = "fahrenheit"
    CELSIUS = "celsius"


class FncCtx(FunctionContext):
    @ai_callable(
        description="Get the current weather in a given location", auto_retry=True
    )
    def get_weather(
        self,
        location: Annotated[
            str, TypeInfo(description="The city and state, e.g. San Francisco, CA")
        ],
        unit: Annotated[
            Unit, TypeInfo(description="The temperature unit to use.")
        ] = Unit.CELSIUS,
    ) -> None: ...

    @ai_callable(description="Play a music")
    def play_music(
        self,
        name: Annotated[str, TypeInfo(description="the name of the Artist")],
    ) -> None: ...

    # test for cancelled calls
    @ai_callable(description="Turn on/off the lights in a room")
    async def toggle_light(
        self,
        room: Annotated[str, TypeInfo(description="The room to control")],
        on: bool = True,
    ) -> None:
        await asyncio.sleep(60)

    # used to test arrays as arguments
    @ai_callable(description="Select currencies of a specific area")
    def select_currencies(
        self,
        currencies: Annotated[
            list[str],
            TypeInfo(
                description="The currencies to select",
                choices=["usd", "eur", "gbp", "jpy", "sek"],
            ),
        ],
    ) -> None: ...

    @ai_callable(description="Update user info")
    def update_user_info(
        self,
        email: Annotated[
            Optional[str], TypeInfo(description="The user address email")
        ] = None,
        name: Annotated[Optional[str], TypeInfo(description="The user name")] = None,
        address: Optional[
            Annotated[str, TypeInfo(description="The user address")]
        ] = None,
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
    pytest.param(lambda: openai.LLM.with_vertex(), id="openai.with_vertex"),
]


@pytest.mark.parametrize("llm_factory", LLMS)
async def test_chat(llm_factory: Callable[[], llm.LLM]):
    input_llm = llm_factory()
    chat_ctx = ChatContext().append(
        text='You are an assistant at a drive-thru restaurant "Live-Burger". Ask the customer what they would like to order.'
    )

    # Anthropic and vertex requires at least one message (system messages don't count)
    chat_ctx.append(
        text="Hello",
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
async def test_basic_fnc_calls(llm_factory: Callable[[], llm.LLM]):
    input_llm = llm_factory()
    fnc_ctx = FncCtx()

    stream = await _request_fnc_call(
        input_llm,
        "What's the weather in San Francisco and what's the weather Paris?",
        fnc_ctx,
    )
    calls = stream.execute_functions()
    await asyncio.gather(*[f.task for f in calls])
    await stream.aclose()
    assert len(calls) == 2, "get_weather should be called twice"


@pytest.mark.parametrize("llm_factory", LLMS)
async def test_function_call_exception_handling(llm_factory: Callable[[], llm.LLM]):
    input_llm = llm_factory()
    fnc_ctx = FncCtx()

    @fnc_ctx.ai_callable(description="Simulate a failure")
    async def failing_function():
        raise RuntimeError("Simulated failure")

    stream = await _request_fnc_call(input_llm, "Call the failing function", fnc_ctx)
    calls = stream.execute_functions()
    await asyncio.gather(*[f.task for f in calls], return_exceptions=True)
    await stream.aclose()

    assert len(calls) == 1
    assert isinstance(calls[0].exception, RuntimeError)
    assert str(calls[0].exception) == "Simulated failure"


@pytest.mark.parametrize("llm_factory", LLMS)
async def test_runtime_addition(llm_factory: Callable[[], llm.LLM]):
    input_llm = llm_factory()
    fnc_ctx = FncCtx()
    called_msg = ""

    @fnc_ctx.ai_callable(description="Show a message on the screen")
    async def show_message(
        message: Annotated[str, TypeInfo(description="The message to show")],
    ):
        nonlocal called_msg
        called_msg = message

    stream = await _request_fnc_call(
        input_llm, "Can you show 'Hello LiveKit!' on the screen?", fnc_ctx
    )
    fns = stream.execute_functions()
    await asyncio.gather(*[f.task for f in fns])
    await stream.aclose()

    assert called_msg == "Hello LiveKit!", "send_message should be called"


@pytest.mark.parametrize("llm_factory", LLMS)
async def test_cancelled_calls(llm_factory: Callable[[], llm.LLM]):
    input_llm = llm_factory()
    fnc_ctx = FncCtx()

    stream = await _request_fnc_call(
        input_llm, "Turn off the lights in the bedroom", fnc_ctx
    )
    calls = stream.execute_functions()
    await asyncio.sleep(0.2)  # wait for the loop executor to start the task

    # don't wait for gather_function_results and directly close (this should cancel the ongoing calls)
    await stream.aclose()

    assert len(calls) == 1
    assert isinstance(
        calls[0].exception, asyncio.CancelledError
    ), "toggle_light should have been cancelled"


@pytest.mark.parametrize("llm_factory", LLMS)
async def test_calls_arrays(llm_factory: Callable[[], llm.LLM]):
    input_llm = llm_factory()
    fnc_ctx = FncCtx()

    stream = await _request_fnc_call(
        input_llm,
        "Can you select all currencies in Europe at once from given choices using function call `select_currencies`?",
        fnc_ctx,
        temperature=0.2,
    )
    calls = stream.execute_functions()
    await asyncio.gather(*[f.task for f in calls])
    await stream.aclose()

    assert len(calls) == 1, "select_currencies should have been called only once"

    call = calls[0]
    currencies = call.call_info.arguments["currencies"]
    assert len(currencies) == 3, "select_currencies should have 3 currencies"
    assert (
        "eur" in currencies and "gbp" in currencies and "sek" in currencies
    ), "select_currencies should have eur, gbp, sek"


@pytest.mark.parametrize("llm_factory", LLMS)
async def test_calls_choices(llm_factory: Callable[[], llm.LLM]):
    input_llm = llm_factory()
    fnc_ctx = FncCtx()

    # test choices on int
    @fnc_ctx.ai_callable(description="Change the volume")
    def change_volume(
        volume: Annotated[
            int, TypeInfo(description="The volume level", choices=[0, 11, 30, 83, 99])
        ],
    ) -> None: ...

    if not input_llm.capabilities.supports_choices_on_int:
        with pytest.raises(APIConnectionError):
            stream = await _request_fnc_call(input_llm, "Set the volume to 30", fnc_ctx)
    else:
        stream = await _request_fnc_call(input_llm, "Set the volume to 30", fnc_ctx)
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
    fnc_ctx = FncCtx()

    stream = await _request_fnc_call(
        input_llm, "Using a tool call update the user info to name Theo", fnc_ctx
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
        llm.ToolChoice(type="function", name="get_weather"),
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
    tool_choice: Union[dict, str, None],
    expected_calls: set,
    llm_factory: Callable[[], llm.LLM],
):
    input_llm = llm_factory()
    fnc_ctx = FncCtx()

    stream = await _request_fnc_call(
        input_llm,
        user_request,
        fnc_ctx,
        tool_choice=tool_choice,
        parallel_tool_calls=True,
    )

    calls = stream.execute_functions()
    await asyncio.gather(*[f.task for f in calls], return_exceptions=True)
    await stream.aclose()
    print(calls)

    call_names = {call.call_info.function_info.name for call in calls}
    if tool_choice == "none" and isinstance(input_llm, anthropic.LLM):
        assert True
    else:
        assert (
            call_names == expected_calls
        ), f"Test '{description}' failed: Expected calls {expected_calls}, but got {call_names}"


async def _request_fnc_call(
    model: llm.LLM,
    request: str,
    fnc_ctx: FncCtx,
    temperature: float | None = None,
    parallel_tool_calls: bool | None = None,
    tool_choice: Union[llm.ToolChoice, Literal["auto", "required", "none"]]
    | None = None,
) -> llm.LLMStream:
    stream = model.chat(
        chat_ctx=ChatContext()
        .append(
            text="You are an helpful assistant. Follow the instructions provided by the user. You can use multiple tool calls at once.",
            role="system",
        )
        .append(text=request, role="user"),
        fnc_ctx=fnc_ctx,
        temperature=temperature,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )

    async for _ in stream:
        pass

    return stream


@pytest.mark.parametrize("llm_factory", LLMS)
async def test_chat_with_images(llm_factory: Callable[[], llm.LLM]):
    input_llm = llm_factory()
    
    chat_ctx = ChatContext().append(
        text="You are an AI assistant that can analyze images.",
        role="system",
    ).append(
        text="Analyze this image",
        images=[llm.ChatImage(image=SAMPLE_IMAGE_URL)],
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
