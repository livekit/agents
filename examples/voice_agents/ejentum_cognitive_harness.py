"""Voice agent that calls the Ejentum cognitive harness mid-conversation.

Demonstrates how to expose a third-party REST harness API as a
function tool the voice agent can invoke when the user asks
something that benefits from structured reasoning (planning,
weighing trade-offs, resisting adversarial framing).

The agent sees one tool: `fetch_cognitive_scaffold(task, mode)`. It
picks the right mode for the user's request, gets back a structured
scaffold, and the LLM threads that scaffold into its spoken response.

Get an Ejentum API key at https://ejentum.com/dashboard (free and
paid tiers available). Set `EJENTUM_API_KEY` in your environment.

Run:
    python ejentum_cognitive_harness.py dev
"""

from __future__ import annotations

import logging
import os
from typing import Literal

import aiohttp
from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from livekit.agents.llm import function_tool
from livekit.plugins import silero

logger = logging.getLogger("ejentum-cognitive-harness")
logger.setLevel(logging.INFO)

load_dotenv()


EJENTUM_URL = "https://ejentum-main-ab125c3.zuplo.app/logicv1/"

HarnessMode = Literal["reasoning", "code", "anti-deception", "memory"]


class CognitiveHarnessAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a thoughtful voice assistant. When the user asks something "
                "that needs structured thinking (planning a migration, weighing "
                "trade-offs, debugging a confusing situation, resisting a leading "
                "question), call the fetch_cognitive_scaffold tool first with the "
                "right mode, then thread the returned scaffold into your spoken "
                "answer. Modes: 'reasoning' for planning and trade-offs, 'code' "
                "for software engineering specifics, 'anti-deception' for "
                "questions where the framing might be pressuring you toward a "
                "specific conclusion, 'memory' for 'what should I remember' style "
                "questions."
            ),
        )

    @function_tool
    async def fetch_cognitive_scaffold(self, task: str, mode: HarnessMode) -> str:
        """Fetch a cognitive scaffold from the Ejentum Logic API.

        Call this before answering tasks that benefit from structured
        thinking. The returned scaffold is short structured text the
        agent reads internally to shape its response.

        Args:
            task: A 1-2 sentence description of the user's task.
            mode: Which harness to use. 'reasoning' for planning,
                'code' for software engineering, 'anti-deception' for
                leading-question detection, 'memory' for retention
                decisions.
        """
        api_key = os.environ.get("EJENTUM_API_KEY")
        if not api_key:
            logger.warning("EJENTUM_API_KEY not set; returning empty scaffold")
            return ""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    EJENTUM_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"query": task, "mode": mode},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    if r.status != 200:
                        body = await r.text()
                        logger.warning("Ejentum API %s: %s", r.status, body[:200])
                        return ""
                    payload = await r.json()
        except Exception:
            logger.exception("Ejentum fetch failed")
            return ""

        if isinstance(payload, list) and payload:
            scaffold = payload[0].get(mode, "")
            logger.info("Scaffold retrieved", extra={"mode": mode, "length": len(scaffold)})
            return scaffold
        return ""


async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt=inference.STT(model="assemblyai/universal-streaming:en"),
        llm=inference.LLM(model="openai/gpt-4o-mini"),
        tts=inference.TTS(model="cartesia/sonic-2:794f9389-aac1-45b6-b726-9d9369183238"),
        vad=silero.VAD.load(),
    )

    await session.start(agent=CognitiveHarnessAgent(), room=ctx.room)
    await session.generate_reply(
        instructions=("Greet the user briefly and ask what they would like to think through."),
    )


if __name__ == "__main__":
    cli.run_app(AgentServer(entrypoint))
