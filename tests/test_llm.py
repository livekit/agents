import asyncio
from enum import Enum
from typing import Annotated

from livekit.agents import llm
from livekit.agents.llm import ChatContext, FunctionContext, TypeInfo, ai_callable
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
        self._select_currency_calls = 0
        self._change_volume_calls = 0

        self._toggle_light_cancelled = False
        self._selected_currencies = None
        self._selected_volume = None

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
    ) -> None:
        self._get_weather_calls += 1

    @ai_callable(description="Play a music")
    def play_music(
        self,
        name: Annotated[
            str, TypeInfo(description="The artist and the name of the song")
        ],
    ) -> None:
        self._play_music_calls += 1

    # test for cancelled calls
    @ai_callable(description="Turn on/off the lights in a room")
    async def toggle_light(
        self,
        room: Annotated[str, TypeInfo(description="The room to control")],
        on: bool = True,
    ) -> None:
        self._toggle_light_calls += 1
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self._toggle_light_cancelled = True

    # used to test arrays as arguments
    @ai_callable(description="Currencies of a specific country")
    def select_currencies(
        self,
        currencies: Annotated[
            list[str],
            TypeInfo(
                description="The currency to select",
                choices=["usd", "eur", "gbp", "jpy", "sek"],
            ),
        ],
    ) -> None:
        self._select_currency_calls += 1
        self._selected_currencies = currencies

    # test choices on int
    @ai_callable(description="Change the volume")
    def change_volume(
        self,
        volume: Annotated[
            int, TypeInfo(description="The volume level", choices=[0, 11, 30, 83, 99])
        ],
    ) -> None:
        self._change_volume_calls += 1
        self._selected_volume = volume


async def test_chat():
    llm = openai.LLM(model="gpt-4o")

    chat_ctx = ChatContext().append(
        text='You are an assistant at a drive-thru restaurant "Live-Burger". Ask the customer what they would like to order.'
    )

    stream = llm.chat(chat_ctx=chat_ctx)
    text = ""
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            text += content

    assert len(text) > 0


async def test_fnc_calls():
    fnc_ctx = FncCtx()
    llm = openai.LLM(model="gpt-4o")

    stream = await _request_fnc_call(
        llm, "What's the weather in San Francisco and Paris?", fnc_ctx
    )
    fns = stream.execute_functions()
    await asyncio.gather(*[f.task for f in fns])
    await stream.aclose()

    assert fnc_ctx._get_weather_calls == 2, "get_weather should be called twice"


async def test_fnc_calls_runtime_addition():
    fnc_ctx = FncCtx()
    llm = openai.LLM(model="gpt-4o")
    called_msg = ""

    @fnc_ctx.ai_callable(description="Show a message on the screen")
    async def show_message(
        message: Annotated[str, TypeInfo(description="The message to show")],
    ):
        nonlocal called_msg
        called_msg = message

    stream = await _request_fnc_call(
        llm, "Can you show 'Hello LiveKit!' on the screen?", fnc_ctx
    )
    fns = stream.execute_functions()
    await asyncio.gather(*[f.task for f in fns])
    await stream.aclose()

    assert called_msg == "Hello LiveKit!", "send_message should be called"


async def test_cancelled_calls():
    fnc_ctx = FncCtx()
    llm = openai.LLM(model="gpt-4o")

    stream = await _request_fnc_call(
        llm, "Turn off the lights in the Theo's bedroom", fnc_ctx
    )
    stream.execute_functions()

    # Need to wait for the task to start
    await asyncio.sleep(0)

    # don't wait for gather_function_results and directly close
    await stream.aclose()

    assert fnc_ctx._toggle_light_calls == 1
    assert fnc_ctx._toggle_light_cancelled, "toggle_light should be cancelled"


async def test_calls_arrays():
    fnc_ctx = FncCtx()
    llm = openai.LLM(model="gpt-4o")

    stream = await _request_fnc_call(
        llm, "Can you select all currencies in Europe?", fnc_ctx
    )
    fns = stream.execute_functions()
    await asyncio.gather(*[f.task for f in fns])
    await stream.aclose()

    assert fnc_ctx._select_currency_calls == 1
    assert fnc_ctx._selected_currencies is not None
    assert len(fnc_ctx._selected_currencies) == 3

    assert "eur" in fnc_ctx._selected_currencies
    assert "gbp" in fnc_ctx._selected_currencies
    assert "sek" in fnc_ctx._selected_currencies


async def test_calls_choices():
    fnc_ctx = FncCtx()
    llm = openai.LLM(model="gpt-4o")

    stream = await _request_fnc_call(llm, "Set the volume to 30", fnc_ctx)
    fns = stream.execute_functions()
    await asyncio.gather(*[f.task for f in fns])
    await stream.aclose()

    assert fnc_ctx._change_volume_calls == 1
    assert fnc_ctx._selected_volume == 30


async def _request_fnc_call(
    model: llm.LLM, request: str, fnc_ctx: FncCtx
) -> llm.LLMStream:
    stream = model.chat(
        chat_ctx=ChatContext().append(text=request, role="user"), fnc_ctx=fnc_ctx
    )

    async for _ in stream:
        pass

    return stream
