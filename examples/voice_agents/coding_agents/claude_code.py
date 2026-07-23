"""Claude Code as a background session — a harness-engineered coding agent.

A voice agent fronts one long-lived Claude Code (Claude Agent SDK) session.
The harness separates what each side sees into explicit channels, so the
voice LLM always has the right context at the right cost:

1. SPEAK — Claude Code calls its ``send_to_user`` tool (an in-process MCP
   tool). The message goes through ``ctx.send()``, which schedules a spoken
   reply: the voice agent relays it to the user (possibly rephrased —
   background sessions never talk to the user directly): questions,
   decisions to confirm, and completion summaries.
2. CONTEXT-ONLY — Claude Code's plain narration text is inserted silently
   (``ctx.send(..., silent=True)``): the voice LLM can see the full working
   narrative without the user having to listen to it.
3. STATE — a live, structured snapshot via ``ctx.set_state()`` (status,
   current task, recent activity, files changed). The voice LLM queries it
   with the generated ``lk_background_state`` tool when the user asks
   "what's going on right now?".
4. INTERRUPT — any message from the voice side while Claude Code is
   mid-turn calls ``client.interrupt()``; the message is then delivered as
   the next user turn, so "stop, do X instead" behaves like a real
   interruption instead of queueing behind the current task.

Requires: pip install claude-agent-sdk (and the ``claude`` CLI installed)

Run with: uv run examples/voice_agents/coding_agents/claude_code.py dev
"""

import asyncio
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
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

logger = logging.getLogger("claude-code-agent")

load_dotenv()

# Claude Code drives the repository this example lives in
REPO_ROOT = Path(__file__).parents[3]

# messages arriving this soon after a turn starts are treated as part of the same
# user utterance (voice LLMs often split one correction into several sends) and are
# queued instead of interrupting the turn they just dispatched
INTERRUPT_GRACE = 2.0

CLAUDE_CODE_PROMPT = """You are pair-programming with a user who is on a live voice call.

Communication channels — follow them strictly:
- Your plain text output is NOT spoken. It is recorded as background context the voice
  assistant can read, so narrate your work there freely.
- To say something the user must hear — a question you need answered, a decision you want
  confirmed, or a short summary when you finish a task — call the send_to_user tool. The
  voice assistant relays it aloud: keep it to one or two conversational sentences, no
  markdown, no code, no file paths unless essential. Ask one question at a time, then end
  your turn and wait for the answer.
- You may be interrupted mid-task. When that happens, the next user message is a
  correction or a new direction — apply it to what you were doing rather than starting over.
"""


@background(name="claude_code")
async def claude_code(ctx: BackgroundContext) -> None:
    """Runs coding tasks with Claude Code in this repository. Send it a task to
    start, answers to its questions, or a new direction at any time — messages
    sent while it is working interrupt it immediately. It speaks its own
    questions and completion summaries, and reports live status (current task,
    recent activity, files changed) through the state tool."""
    files_changed: list[str] = []
    activity: deque[str] = deque(maxlen=8)
    status: dict[str, Any] = {"status": "idle"}

    def report(**changes: Any) -> None:
        status.update(changes)
        ctx.set_state(
            {
                **status,
                "recent_activity": list(activity),
                # drop files that have since been deleted so state never reports
                # a change that no longer exists on disk
                "files_changed": [p for p in files_changed if Path(p).exists()],
            }
        )

    # SPEAK channel: Claude Code pushes voice updates itself. By design a
    # background session never talks to the user directly — ctx.send() schedules
    # a spoken reply that the voice agent delivers (and may lightly rephrase).
    @tool(
        "send_to_user",
        "Send a message for the voice assistant to relay to the user. Use this for "
        "questions you need answered, decisions you want confirmed, and short "
        "completion summaries. One or two plain conversational sentences; no markdown.",
        {"message": str},
    )
    async def send_to_user(args: dict[str, Any]) -> dict[str, Any]:
        await ctx.send(str(args["message"]))
        return {"content": [{"type": "text", "text": "Delivered to the user."}]}

    voice_server = create_sdk_mcp_server(name="voice", version="1.0.0", tools=[send_to_user])

    options = ClaudeAgentOptions(
        system_prompt=CLAUDE_CODE_PROMPT,
        mcp_servers={"voice": voice_server},
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "mcp__voice__send_to_user"],
        permission_mode="acceptEdits",
        cwd=str(REPO_ROOT),
    )

    inbox: asyncio.Queue[str] = asyncio.Queue()
    working = False
    turn_started = 0.0

    async with ClaudeSDKClient(options=options) as client:

        async def _read_inbox() -> None:
            async for message in ctx.message_stream():
                # INTERRUPT channel: the user spoke while Claude Code is mid-turn —
                # stop it; the message becomes the next turn. Guarded so a burst of
                # split messages interrupts once: later pieces just queue behind the
                # first instead of killing the correction turn they belong to.
                if working and inbox.empty() and time.monotonic() - turn_started > INTERRUPT_GRACE:
                    try:
                        await client.interrupt()
                        report(status="interrupted, switching to the user's new instruction")
                    except Exception:
                        logger.exception("failed to interrupt claude code")
                inbox.put_nowait(message)

        reader = asyncio.create_task(_read_inbox())
        try:
            report(status="idle, waiting for a task")
            while True:
                message = await inbox.get()
                working = True
                turn_started = time.monotonic()
                report(status="working", task=message[:150])
                await client.query(message)

                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock) and block.text.strip():
                                # CONTEXT-ONLY channel: narration enters the voice
                                # agent's context without being spoken.
                                await ctx.send(block.text, silent=True)
                            elif isinstance(block, ToolUseBlock):
                                if block.name in ("Write", "Edit"):
                                    path = str(block.input.get("file_path", ""))
                                    if path and path not in files_changed:
                                        files_changed.append(path)
                                    activity.append(f"edited {path}")
                                elif block.name == "Bash":
                                    activity.append(
                                        f"ran `{str(block.input.get('command', ''))[:60]}`"
                                    )
                                elif block.name in ("Glob", "Grep"):
                                    continue  # search churn — too noisy for state
                                report()  # STATE channel: live after every action
                    elif isinstance(msg, ResultMessage):
                        logger.info("claude code turn done", extra={"subtype": msg.subtype})

                working = False
                report(status="idle, waiting for the next task", last_task=message[:150])
        finally:
            reader.cancel()


class CodingAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a voice coding assistant fronting a Claude Code session running "
                "in the background.\n"
                "- Send the user's coding tasks to the claude_code background session.\n"
                "- Sending a task only delivers it — it does NOT mean the work is done. After "
                "dispatching, say you've passed it along and nothing more. Never claim a file "
                "was created, changed, or deleted until Claude Code itself announces it.\n"
                "- Claude Code speaks its own questions and completion summaries; relay its "
                "updates naturally and forward the user's answers back to it immediately.\n"
                "- Forward corrections or new directions right away — if Claude Code is "
                "mid-task, the message interrupts it automatically.\n"
                "- If the user asks what is happening right now, call the lk_background_state "
                "tool for the live status (current task, recent activity, files changed) and "
                "answer from it. Its silent context notes also show the working narrative.\n"
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
