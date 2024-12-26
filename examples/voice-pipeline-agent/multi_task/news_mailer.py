import asyncio
import json
import logging
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import AgentCallContext, VoicePipelineAgent
from livekit.agents.pipeline.agent_task import AgentTask
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.plugins import deepgram, openai, silero

load_dotenv()

logger = logging.getLogger("news-mailer")
logger.setLevel(logging.INFO)


class UserData(TypedDict):
    query: str | None
    news: str | None
    email: str | None


@llm.ai_callable()
async def query_news(
    query: Annotated[str, llm.TypeInfo(description="The query user asked for")],
) -> str:
    """Called to query news from the internet.
    Tell the user you are checking the news when calling this function."""
    logger.info(f"Querying news for {query}")
    perplexity_llm = openai.LLM.with_perplexity(
        model="llama-3.1-sonar-small-128k-online"
    )
    chat_ctx = llm.ChatContext().append(
        role="system",
        text="Search the recent news articles about the query.",
    )
    chat_ctx.append(role="user", text=query)
    llm_stream = perplexity_llm.chat(chat_ctx=chat_ctx)
    news = ""
    async for chunk in llm_stream:
        if not chunk or not chunk.choices or not chunk.choices[0].delta.content:
            continue
        news += chunk.choices[0].delta.content

    agent = AgentCallContext.get_current().agent
    user_data: UserData = agent.user_data
    user_data["query"] = query
    user_data["news"] = news
    logger.info(f"The news about {query} collected")
    return news


@llm.ai_callable()
async def send_news_email() -> str:
    """Called to send the news to the user's email address."""
    agent = AgentCallContext.get_current().agent
    user_data: UserData = agent.user_data
    email = user_data.get("email")
    news = user_data.get("news")

    if not email:
        return "email is not collected"

    if not news:
        return "news is not collected"

    # mock sending email
    query = user_data.get("query")
    logger.info(f"Sending news about {query} to {email}")
    await asyncio.sleep(2)
    return f"The news about {query} is sent to {email}"


@llm.ai_callable()
async def verify_email(
    email: Annotated[str, llm.TypeInfo(description="The collected email address")],
) -> str:
    """Called to verify the user's email address."""
    if "@" not in email:
        return "The email address is not valid, please confirm with the user."

    # Potentially show the email on the screen for the user to confirm
    return "The email address is valid. Please confirm with the user for the spelling."


@llm.ai_callable()
async def update_email(
    email: Annotated[str, llm.TypeInfo(description="The collected email address")],
) -> str:
    """Called to update the user's email address."""

    agent = AgentCallContext.get_current().agent
    user_data: UserData = agent.user_data
    user_data["email"] = email
    logger.info(f"The email is updated to {email}")
    return f"The email is updated to {email}."


news_mailer = AgentTask(
    name="news_mailer",
    instructions=(
        "You are a friendly assistant that can query news from the internet."
        "Summarize the news in 50 words or less and ask the user if they want to receive the news by email."
        "Use email_collector to collect the user's email address."
    ),
    functions=[query_news, send_news_email],
)

email_collector = AgentTask(
    name="email_collector",
    instructions=(
        "You are a friendly assistant that can collect the user's email address. Your tasks:\n"
        "1. Collect the user's email address, help to complete the @ and domain part if possible.\n"
        "2. Verify the address with `verify_email` function until the user confirms.\n"
        "3. Update the email address after the user confirms.\n"
        "Transfer back to news_mailer after the email is updated."
    ),
    functions=[update_email, verify_email],
)


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    chat_log_file = "news_mailer.log"

    # Set up chat logger
    chat_logger = logging.getLogger("chat_logger")
    chat_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(chat_log_file)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    chat_logger.addHandler(handler)

    participant = await ctx.wait_for_participant()
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        initial_task=news_mailer,
        available_tasks=[news_mailer, email_collector],
        max_nested_fnc_calls=3,  # may call functions in the transition function
    )

    # read text input from the room for easy testing
    @ctx.room.on("data_received")
    def on_data_received(packet: rtc.DataPacket):
        if packet.topic == "lk-chat-topic":
            data = json.loads(packet.data.decode("utf-8"))
            logger.debug("Text input received", extra={"message": data["message"]})

            agent._human_input.emit(
                "final_transcript",
                SpeechEvent(
                    type=SpeechEventType.END_OF_SPEECH,
                    alternatives=[SpeechData(language="en", text=data["message"])],
                ),
            )

    # write the chat log to a file
    @agent.on("user_speech_committed")
    @agent.on("agent_speech_interrupted")
    @agent.on("agent_speech_committed")
    def on_speech_committed(message: llm.ChatMessage):
        chat_logger.info(f"{message.role}: {message.content}")

    @agent.on("function_calls_collected")
    def on_function_calls_collected(calls: list[llm.FunctionCallInfo]):
        fnc_infos = [{fnc.function_info.name: fnc.arguments} for fnc in calls]
        chat_logger.info(f"fnc_calls_collected: {fnc_infos}")

    @agent.on("function_calls_finished")
    def on_function_calls_finished(calls: list[llm.CalledFunction]):
        called_infos = [{fnc.call_info.function_info.name: fnc.result} for fnc in calls]
        chat_logger.info(f"fnc_calls_finished: {called_infos}")

    # Start the assistant. This will automatically publish a microphone track and listen to the participant.
    agent.start(ctx.room, participant)
    await agent.say("Welcome to news mailer! How may I assist you today?")


def prewarm_process(proc: JobProcess):
    # preload silero VAD in memory to speed up session start
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm_process,
        ),
    )
