# Copyright 2026 LiveKit, Inc.
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

"""DashScope voice agent.

Set DASHSCOPE_API_KEY and LiveKit credentials, then run:
    uv run python examples/voice_agents/alibaba_realtime.py dev
"""

from datetime import datetime, timezone

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, function_tool
from livekit.plugins import alibaba

load_dotenv()
server = AgentServer()


@function_tool
async def current_time() -> str:
    """Return the current UTC date and time."""
    return datetime.now(timezone.utc).isoformat()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """Start a speech-to-speech session with one local tool."""
    session: AgentSession[None] = AgentSession(llm=alibaba.realtime.RealtimeModel())
    await session.start(
        agent=Agent(
            instructions="You are a helpful voice assistant. Keep answers brief.",
            tools=[current_time],
        ),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
