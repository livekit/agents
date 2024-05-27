import asyncio
from enum import Enum
from typing import Annotated

from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    ChatRole,
    FunctionContext,
    TypeInfo,
    ai_callable,
)
from livekit.plugins import openai


class Unit(Enum):
    FAHRENHEIT = "fahrenheit"
    CELSIUS = "celsius"


class FncCtx(FunctionContext):
    def __init__(self) -> None:
        super().__init__()
        self._get_weather_calls = 0
        self._play_music_calls = 0
        self._toggle_light_calls = 0
        self._toggle_light_cancelled = False

    @ai_callable(desc="Get the current weather in a given location", auto_retry=True)
    def get_weather(
        self,
        location: Annotated[
            str, TypeInfo(desc="The city and state, e.g. San Francisco, CA")
        ],
        unit: Annotated[
            Unit,
            TypeInfo(desc="The temperature unit to use."),
        ] = Unit.CELSIUS,
    ) -> None:
        self._get_weather_calls += 1

    @ai_callable(desc="Play a music")
    def play_music(
        self,
        name: Annotated[str, TypeInfo(desc="The artist and the name of the song")],
    ) -> None:
        self._play_music_calls += 1

    @ai_callable(desc="Turn on/off the lights in a room")
    async def toggle_light(
        self,
        room: Annotated[str, TypeInfo(desc="The room to control")],
        on: bool = True,
    ) -> None:
        self._toggle_light_calls += 1
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self._toggle_light_cancelled = True


async def test_chat():
    llm = openai.LLM(model="gpt-4o")

    chat_ctx = ChatContext(
        messages=[
            ChatMessage(
                role=ChatRole.SYSTEM,
                text=(
                    'You are an assistant at a drive-thru restaurant "Live-Burger". '
                    "Ask the customer what they would like to order."
                ),
            ),
        ]
    )

    stream = await llm.chat(chat_ctx=chat_ctx)
    text = ""
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            text += content

    assert len(text) > 0


async def test_fnc_calls():
    fnc_ctx = FncCtx()
    llm = openai.LLM(model="gpt-4o")

    # test fnc calls
    stream = await llm.chat(
        chat_ctx=ChatContext(
            messages=[
                ChatMessage(
                    role=ChatRole.USER,
                    text="What's the weather in San Francisco and Paris?",
                ),
            ]
        ),
        fnc_ctx=fnc_ctx,
    )

    async for _ in stream:
        pass

    await stream.aclose()

    assert fnc_ctx._get_weather_calls == 2, "get_weather should be called twice"


async def test_fnc_calls_runtime_addition():
    fnc_ctx = FncCtx()
    llm = openai.LLM(model="gpt-4o")
    called_msg = ""

    @fnc_ctx.ai_callable(desc="Show a message on the screen")
    async def show_message(
        message: Annotated[
            str,
            TypeInfo(
                desc="The message to show",
            ),
        ],
    ):
        nonlocal called_msg
        called_msg = message

    # test fnc calls
    stream = await llm.chat(
        chat_ctx=ChatContext(
            messages=[
                ChatMessage(
                    role=ChatRole.USER,
                    text='Can you show "Hello LiveKit!" on the screen?',
                ),
            ]
        ),
        fnc_ctx=fnc_ctx,
    )

    async for _ in stream:
        pass

    await stream.aclose()

    assert called_msg == "Hello LiveKit!", "send_message should be called"


async def test_cancelled_calls():
    fnc_ctx = FncCtx()
    llm = openai.LLM(model="gpt-4o")

    stream = await llm.chat(
        chat_ctx=ChatContext(
            messages=[
                ChatMessage(
                    role=ChatRole.USER,
                    text="Turn off the lights in the Theo's bedroom",
                ),
            ]
        ),
        fnc_ctx=fnc_ctx,
    )

    async for _ in stream:
        pass

    await stream.aclose(wait=False)  # cancel running function calls

    assert fnc_ctx._toggle_light_calls == 1
    assert fnc_ctx._toggle_light_cancelled, "toggle_light should be cancelled"
