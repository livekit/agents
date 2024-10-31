import logging
import pickle
from datetime import datetime
from typing import Annotated

import pandas as pd
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, rag, silero

load_dotenv()

logger = logging.getLogger("weather-demo")
logger.setLevel(logging.INFO)

annoy_index = rag.annoy.AnnoyIndex.load("vdb_data")  # see build_data.py

embeddings_dimension = 1536
with open("my_data.pkl", "rb") as f:
    paragraphs_by_uuid = pickle.load(f)


class AssistantFnc(llm.FunctionContext):
    """
    The class defines a set of LLM functions that the assistant can execute.
    """

    @llm.ai_callable()
    async def check_reservation(
        self,
        date: Annotated[
            str,
            llm.TypeInfo(
                description="The date and time to see whether a reservation is available"
            ),
        ],
    ):
        """Called when the customer provides a date and time for reservation enquire. This function will return whether reservation is available on the provided date and time, or other alternate availabilities."""

        logger.info(f"checking reservation on {date}")

        df = pd.read_csv("dates.csv")

        # formatting datetime to query csv
        parsed_datetime = datetime.fromisoformat(date)
        date_to_check = parsed_datetime.strftime("%Y-%m-%d")
        time_to_check = parsed_datetime.strftime("%H:%M")

        result = df[(df["Date"] == date_to_check) & (df["Time Slot"] == time_to_check)]

        print("result", result)

        # find other available slots on the same date
        available_slots = df[
            (df["Date"] == date_to_check) & (df["Is Reserved"] == "Not Reserved")
        ]

        if result.empty:
            if not available_slots.empty:
                print("Other available slots on this date:")
                print(available_slots["Time Slot"].to_string(index=False))
                return f"The reservation is not available for {date}. Other available slots on this date: {available_slots["Time Slot"].to_string(index=False)}"
            else:
                print("No other available slots on this date.")
                return f"The reservation is not available for {date}, and there is no alternate options."
        elif result["Is Reserved"].values[0] == "Reserved":
            print(f"The slot on {date_to_check} at {time_to_check} is Reserved.")

            if not available_slots.empty:
                print("Other available slots on this date:")
                print(available_slots["Time Slot"].to_string(index=False))
                return f"The reservation is not available for {date}. Other available slots on this date: {available_slots["Time Slot"].to_string(index=False)}"
            else:
                print("No other available slots on this date.")
                return f"The reservation is not available for {date}, and there is no alternate options."
        else:
            return f"The reservation is available for {date}"

    @llm.ai_callable()
    async def make_booking(
        self,
        date: Annotated[
            str,
            llm.TypeInfo(description="The date and time to create a reservation"),
        ],
    ):
        """Called when the customer confirms a date and time to a spot at the restaurant. This function will return a message saying reservation has been made."""

        logger.info(f"creating reservation on {date}")

        df = pd.read_csv("dates.csv")

        parsed_datetime = datetime.fromisoformat(date)
        date_to_check = parsed_datetime.strftime("%Y-%m-%d")
        time_to_check = parsed_datetime.strftime("%H:%M")

        # update csv to mark requested date and time reserved
        df.loc[
            (df["Date"] == date_to_check) & (df["Time Slot"] == time_to_check),
            "Is Reserved",
        ] = "Reserved"
        df.to_csv("dates.csv", index=False)

        return f"The reservation has been made for {date}."


def prewarm_process(proc: JobProcess):
    # preload silero VAD in memory to speed up session start
    proc.userdata["vad"] = silero.VAD.load()


PROMPT = """
You are a supervisor agent Sphinx Restaurant. You have two assistants, Frontdesk and Booking. Each assistant is associated with a function call.

Assistant responsibilities:
- Frontdesk: Validates availability and handles customer questions related to the restaurant. Only use the provided context if the customer asks restaurant related questions. The check_reservation function will be called when a customer gives a date and time for reservation. Assume that all date is in 2024. If requested timeslot is not available but other timeslots are available, make sure to only list a maximum of two or three options from the available options, and not more than that. Only provide alternate options if provided, otherwise request for a new date and time.
- Booking: Creates final reservation booking after availability is confirmed. Before making the booking, the supervisor should get confirmation from the customer one last time by repeating the intended date and time for reservation. The make_booking function should be called when a reservation is available and is confirmed by the customer.

Workflow rules:
1. Frontdesk must check availability first. Availability requires both date and time.
2. Booking can only process after availability is confirmed. Make sure customer confirmation is received before making the booking.
3. Conclude the call when:
   - Customer's request is completed or question is answered
   - Booking is successfully completed
"""


async def entrypoint(ctx: JobContext):

    async def _enrich_with_rag(agent: VoicePipelineAgent, chat_ctx: llm.ChatContext):
        # locate the last user message and use it to query the RAG model
        # to get the most relevant paragraph
        # then provide that as additional context to the LLM
        user_msg = chat_ctx.messages[-1]
        user_embedding = await openai.create_embeddings(
            input=[user_msg.content],
            model="text-embedding-3-small",
            dimensions=embeddings_dimension,
        )

        result = annoy_index.query(user_embedding[0].embedding, n=1)[0]
        paragraph = paragraphs_by_uuid[result.userdata]
        if paragraph:
            logger.info(f"enriching with RAG: {paragraph}")
            rag_msg = llm.ChatMessage.create(
                text="Context:\n" + paragraph,
                role="assistant",
            )
            # replace last message with RAG, and append user message at the end
            chat_ctx.messages[-1] = rag_msg
            chat_ctx.messages.append(user_msg)

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    fnc_ctx = AssistantFnc()  # create our fnc ctx instance
    initial_chat_ctx = llm.ChatContext().append(
        text=(PROMPT),
        role="system",
    )
    participant = await ctx.wait_for_participant()
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        fnc_ctx=fnc_ctx,
        chat_ctx=initial_chat_ctx,
        before_llm_cb=_enrich_with_rag,
    )

    agent.start(ctx.room, participant)
    await agent.say("Welcome to Sphinx restaurant. How may I assist you?.")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm_process,
        ),
    )
