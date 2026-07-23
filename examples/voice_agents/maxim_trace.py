import logging
import os
import uuid

import dotenv
from livekit import agents
from livekit import api as livekit_api
from livekit.agents import Agent, AgentSession, function_tool
from livekit.api.room_service import CreateRoomRequest
from livekit.plugins import openai
from maxim import Maxim
from maxim.logger.livekit import instrument_livekit
from tavily import TavilyClient

# Load environment variables
dotenv.load_dotenv(override=True)
logging.basicConfig(level=logging.DEBUG)

"""
This example shows how to use the maxim tracer to trace the agent session.
To run this agent example, set MAXIM_API_KEY, MAXIM_LOG_REPO_ID, TAVILY_API_KEY & OPENAI_API_KEY environment variables.
To install maxim, run `pip install maxim-py`
"""

logger = Maxim().logger()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# Maxim event instrumentation
def on_event(event: str, data: dict):
    if event == "maxim.trace.started":
        trace_id = data["trace_id"]
        trace = data["trace"]
        logging.debug(f"Trace started - ID: {trace_id}", extra={"trace": trace})
    elif event == "maxim.trace.ended":
        trace_id = data["trace_id"]
        trace = data["trace"]
        logging.debug(f"Trace ended - ID: {trace_id}", extra={"trace": trace})


instrument_livekit(logger, on_event)


class InterviewAgent(Agent):
    def __init__(self, jd: str) -> None:
        super().__init__(
            instructions=f"You are a professional interviewer. The job description is: {jd}\nAsk relevant interview questions, listen to answers, and follow up as a real interviewer would."
        )

    @function_tool()
    async def web_search(self, query: str) -> str:
        """
        Performs a web search for the given query.
        """
        if not TAVILY_API_KEY:
            return "Tavily API key is not set. Please set the TAVILY_API_KEY environment variable."
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        try:
            response = tavily_client.search(query=query, search_depth="basic")
            if response.get("answer"):
                return response["answer"]
            return str(response.get("results", "No results found."))
        except Exception as e:
            return f"An error occurred during web search: {e}"


async def entrypoint(ctx: agents.JobContext):
    # Prompt user for JD at the start
    jd = input("Paste the Job Description (JD) and press Enter:\n")
    room_name = os.getenv("LIVEKIT_ROOM_NAME") or f"interview-room-{uuid.uuid4().hex}"
    lkapi = livekit_api.LiveKitAPI(
        url=os.getenv("LIVEKIT_URL"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    )
    try:
        req = CreateRoomRequest(
            name=room_name,
            empty_timeout=600,  # keep the room alive 10m after empty
            max_participants=2,  # interviewer + candidate
        )
        room = await lkapi.room.create_room(req)
        print(f"Room created: {room}")
        session = AgentSession(
            llm=openai.realtime.RealtimeModel(voice="coral"),
        )
        await session.start(room=room, agent=InterviewAgent(jd))
        await ctx.connect()
        await session.generate_reply(
            instructions="Greet the candidate and start the interview."
        )
    finally:
        await lkapi.aclose()


if __name__ == "__main__":
    opts = agents.WorkerOptions(entrypoint_fnc=entrypoint)
    agents.cli.run_app(opts)
