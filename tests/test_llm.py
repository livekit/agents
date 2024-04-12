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


async def test_fnc_calls():
    fnc_ctx = FncCtx()
    llm = openai.LLM(model="gpt-4-1106-preview")

    # test fnc calls
    stream = await llm.chat(
        history=ChatContext(
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

    assert fnc_ctx._get_weather_calls == 2

    # test cancellable
    stream = await llm.chat(
        history=ChatContext(
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
    assert fnc_ctx._toggle_light_cancelled
