"""Claude Code as a background session — simplified example.

A voice agent fronts a long-lived Claude Code (Claude Agent SDK) session
running as a ``@background`` session:

    claude code -- ctx.send(progress / results / questions) --> voice agent --> user
    user        -- voice answer --> voice agent -- lk_background_send --> claude code

The Claude Agent SDK keeps one interactive session alive for the whole voice
call: every message the user relays becomes a follow-up ``client.query()``
with full context retained, and Claude's streamed responses are forwarded to
the voice conversation as asynchronous updates.

Requires: pip install claude-agent-sdk (and the ``claude`` CLI installed)

Run with: uv run examples/voice_agents/background_claude_code.py dev
"""

import logging
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    BackgroundContext,
    JobContext,
    background,
    cli,
    inference,
)

logger = logging.getLogger("background-claude-code")

load_dotenv()


@background(name="claude_code")
async def claude_code(ctx: BackgroundContext) -> None:
    """Runs coding tasks with Claude Code in this repository. Send it a task
    to start; it streams progress back and may ask questions — relay them to
    the user and send the answers back. It keeps full session context, so
    follow-ups and course corrections can be sent at any time."""
    options = ClaudeAgentOptions(
        system_prompt=(
            "You are pair-programming with a user who is on a voice call. Your text "
            "output is read aloud to them, so narrate progress in short plain sentences "
            "and ask questions one at a time when you need input."
        ),
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="acceptEdits",
        cwd=str(Path(__file__).parent),
    )
    async with ClaudeSDKClient(options=options) as client:
        files_changed: list[str] = []
        ctx.set_state({"status": "idle", "files_changed": files_changed})
        async for message in ctx.message_stream():  # voice -> claude code (FIFO)
            ctx.set_state(
                {"status": "working", "task": message[:150], "files_changed": files_changed}
            )
            await client.query(message)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock) and block.text.strip():
                            await ctx.send(block.text)  # claude code -> voice
                        elif isinstance(block, ToolUseBlock) and block.name in ("Write", "Edit"):
                            path = str(block.input.get("file_path", ""))
                            if path and path not in files_changed:
                                files_changed.append(path)
                                # context-only: the voice agent knows without speaking it
                                await ctx.send(f"Claude Code modified {path}", silent=True)
                elif isinstance(msg, ResultMessage):
                    logger.info("claude code turn done", extra={"subtype": msg.subtype})
            ctx.set_state(
                {"status": "idle", "last_task": message[:150], "files_changed": files_changed}
            )


class CodingAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a voice coding assistant fronting a Claude Code session running "
                "in the background.\n"
                "- Send the user's coding tasks to the claude_code background session.\n"
                "- Relay its progress updates and questions conversationally; forward the "
                "user's answers and follow-ups back to it immediately.\n"
                "- If the user asks what is happening right now, call the lk_background_state "
                "tool for the real-time status and answer from it.\n"
                "- Keep responses concise; no markdown or special characters."
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Greet the user and ask what to build or fix.")


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session: AgentSession = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        background=[claude_code],
    )
    await session.start(agent=CodingAssistant(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
