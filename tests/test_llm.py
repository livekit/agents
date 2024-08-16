from __future__ import annotations

import asyncio
from enum import Enum
from typing import Annotated, Optional

from livekit.agents import llm
from livekit.agents.llm import ChatContext, FunctionContext, TypeInfo, ai_callable
from livekit.plugins import openai


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
        name: Annotated[
            str, TypeInfo(description="The artist and the name of the song")
        ],
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
    @ai_callable(description="Currencies of a specific area")
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

    # test choices on int
    @ai_callable(description="Change the volume")
    def change_volume(
        self,
        volume: Annotated[
            int, TypeInfo(description="The volume level", choices=[0, 11, 30, 83, 99])
        ],
    ) -> None: ...

    @ai_callable(description="Update userinfo")
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


async def test_basic_fnc_calls():
    fnc_ctx = FncCtx()
    llm = openai.LLM(model="gpt-4o")

    stream = await _request_fnc_call(
        llm, "What's the weather in San Francisco and Paris?", fnc_ctx
    )
    calls = stream.execute_functions()
    await asyncio.gather(*[f.task for f in calls])
    await stream.aclose()
    assert len(calls) == 2, "get_weather should be called twice"


async def test_runtime_addition():
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
    calls = stream.execute_functions()
    await asyncio.sleep(0)  # wait for the loop executor to start the task

    # don't wait for gather_function_results and directly close (this should cancel the ongoing calls)
    await stream.aclose()

    assert len(calls) == 1
    assert isinstance(
        calls[0].exception, asyncio.CancelledError
    ), "toggle_light should have been cancelled"


async def test_calls_arrays():
    fnc_ctx = FncCtx()
    llm = openai.LLM(model="gpt-4o")

    stream = await _request_fnc_call(
        llm,
        "Can you select all currencies in Europe at once?",
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


async def test_calls_choices():
    fnc_ctx = FncCtx()
    llm = openai.LLM(model="gpt-4o")

    stream = await _request_fnc_call(llm, "Set the volume to 30", fnc_ctx)
    calls = stream.execute_functions()
    await asyncio.gather(*[f.task for f in calls])
    await stream.aclose()

    assert len(calls) == 1, "change_volume should have been called only once"

    call = calls[0]
    volume = call.call_info.arguments["volume"]
    assert volume == 30, "change_volume should have been called with volume 30"


async def test_optional_args():
    fnc_ctx = FncCtx()
    llm = openai.LLM(model="gpt-4o")

    stream = await _request_fnc_call(
        llm, "Can you update my information? My name is Theo", fnc_ctx
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


async def _request_fnc_call(
    model: llm.LLM,
    request: str,
    fnc_ctx: FncCtx,
    temperature: float | None = None,
) -> llm.LLMStream:
    stream = model.chat(
        chat_ctx=ChatContext().append(text=request, role="user"),
        fnc_ctx=fnc_ctx,
        temperature=temperature,
    )

    async for _ in stream:
        pass

    return stream
