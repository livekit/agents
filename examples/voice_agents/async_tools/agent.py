# Copyright 2025 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example demonstrating AsyncAgent with @async_function_tool.

AsyncAgent extends Agent and supports async generator tools:
- First yield = immediate tool return (sent to LLM right away)
- Subsequent yields = background notifications with configurable delivery mode
- reply_mode: "when_idle" (default), "interrupt", or "silent"

Run with:
    python examples/voice_agents/async_tools/agent.py dev
"""

import asyncio
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from async_agent import AsyncAgent
from async_function_tool import async_function_tool, notify
from dotenv import load_dotenv

from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    inference,
)
from livekit.plugins import silero

logger = logging.getLogger("async-tools-example")
logger.setLevel(logging.INFO)

load_dotenv()


class ResearchAssistant(AsyncAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful research assistant.",
        )

    @async_function_tool(reply_mode="when_idle")
    async def search_web(self, query: str, run_ctx: RunContext):
        """Search the web for information. Takes a while to get full results.
        Args:
            query: What to search for.
        """
        # First yield = immediate tool return (agent can talk about this right away)
        yield f"Searching for '{query}'... I'll let you know when results are ready."

        # Background: simulate slow API call
        await asyncio.sleep(10)
        results = f"Found 3 articles about '{query}'"

        # Final yield = delivered to agent when it's idle (reply_mode="when_idle")
        yield results

    @async_function_tool(reply_mode="interrupt")
    async def place_order(self, item: str, run_ctx: RunContext):
        """Place an order for an item. Notifies immediately when confirmed.
        Args:
            item: The item to order.
        """
        yield f"Placing order for {item}..."

        # Simulate payment processing
        await asyncio.sleep(10)

        # Final yield with "interrupt" mode = interrupts agent speech to deliver this
        yield f"Order confirmed! {item} will arrive tomorrow."

    @async_function_tool(reply_mode="when_idle")
    async def generate_report(self, topic: str, run_ctx: RunContext):
        """Generate a detailed report with progress updates.
        Args:
            topic: The report topic.
        """
        yield f"Starting report on '{topic}'..."

        for i in range(30):
            await asyncio.sleep(2)
            # Silent notification — just updates context, agent won't speak about it
            yield notify(f"Report progress: section {i + 1}/30 done", "silent")

        # Final yield uses the tool's default reply_mode ("when_idle")
        yield f"Report on '{topic}' is complete with 3 sections."

    async def on_enter(self) -> None:
        self.session.generate_reply(user_input="Greet the user")


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        vad=silero.VAD.load(),
    )

    await session.start(agent=ResearchAssistant(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
