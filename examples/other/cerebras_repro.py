import asyncio

from livekit.agents import function_tool, RunContext, ChatContext, FunctionCall, FunctionCallOutput
from livekit.plugins import openai


@function_tool
async def lookup_weather(
    context: RunContext,
    location: str,
    latitude: str,
    longitude: str,
):
    """Called when the user asks for weather related information.
    Ensure the user's location (city or region) is provided.
    When given a location, please estimate the latitude and longitude of the location and
    do not ask the user for them.

    Args:
        location: The location they are asking for
        latitude: The latitude of the location
        longitude: The longitude of the location
    """

    print(f"Looking up weather for {location}")

    return f"weather for {location} is sunny, 70f"


async def main():
    # when using llama 4 scout, it returns "'NoneType' object is not iterable"
    #llm = openai.LLM.with_cerebras(model="llama-4-scout-17b-16e-instruct")

    # when using llama 3.1b, it behaves incorrectly in the following ways:
    # - sometimes it returns received: id='chatcmpl-8680fa0f-7683-4f78-a476-835e49474a94' delta=ChoiceDelta(role='assistant', content='FN_CALL=False', tool_calls=[])
    # - othertimes it would request the same tool call again repeatedly
    llm = openai.LLM.with_cerebras(model="llama3.1-8b")

    # when using llama 3.3-70b, it would request the same tool call again repeatedly
    llm = openai.LLM.with_cerebras(model="llama-3.3-70b")

    # for the correct result, using OpenAI LLM, it would generate a correct text content given the tool call result
    # llm = openai.LLM()

    ctx = ChatContext()
    ctx.add_message(
        role="system",
        content=(
            "Your name is Kelly. You would interact with users via voice."
            "with that in mind keep your responses concise and to the point."
            "You are curious and friendly, and have a sense of humor."
        ),
    )
    ctx.add_message(role="user", content="How is the weather in San Francisco?")

    stream = llm.chat(
        chat_ctx=ctx,
        tools=[lookup_weather],
    )

    tool_call = None
    async for event in stream:
        print(f"received: {event}")
        if event.delta and event.delta.tool_calls:
            tool_call = event.delta.tool_calls[0]
            ctx.items.append(
                FunctionCall(
                    call_id=tool_call.call_id,
                    arguments=tool_call.arguments,
                    name=tool_call.name,
                )
            )

    result = FunctionCallOutput(
        call_id=tool_call.call_id,
        output=str(await lookup_weather(context=None, location="san francisco", latitude="", longitude="")),
        name=tool_call.name,
        is_error=False,
    )
    print("appending tool call result", result)
    ctx.items.append(result)

    stream = llm.chat(
        chat_ctx=ctx,
        tools=[lookup_weather],
    )
    async for event in stream:
        print(f"received: {event}")

if __name__ == "__main__":
    asyncio.run(main())
