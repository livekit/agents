"""BrowserAgent — high-level AI agent that controls a browser via Anthropic computer-use."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import TYPE_CHECKING

from livekit import rtc
from livekit.agents import llm
from livekit.browser import BrowserContext, BrowserPage  # type: ignore[import-untyped]

from .session import BrowserSession

if TYPE_CHECKING:
    from livekit.plugins.anthropic.computer_tool import ComputerTool

logger = logging.getLogger(__name__)

# Data channel topics
_CHAT_TOPIC = "agent-chat"
_STATUS_TOPIC = "agent-status"


class BrowserAgent:
    """Batteries-included agent that connects an LLM with computer-use to a browser.

    Usage::

        agent = BrowserAgent(
            url="https://www.google.com/",
            llm=LLM(model="claude-sonnet-4-6-20250514"),
            instructions="You are a helpful AI that can browse the web.",
        )
        await agent.start(room=ctx.room)
    """

    def __init__(
        self,
        *,
        url: str = "https://www.google.com/",
        llm: llm.LLM,
        instructions: str = "You are a helpful AI assistant that can browse the web. Use the computer tool to interact with the browser.",
        width: int = 1280,
        height: int = 720,
        framerate: int = 30,
        tools: list[llm.Tool] | None = None,
        chat_enabled: bool = True,
    ) -> None:
        self._url = url
        self._llm = llm
        self._instructions = instructions
        self._width = width
        self._height = height
        self._framerate = framerate
        self._extra_tools = tools or []
        self._chat_enabled = chat_enabled

        self._room: rtc.Room | None = None
        self._browser_ctx: BrowserContext | None = None
        self._page: BrowserPage | None = None
        self._session: BrowserSession | None = None
        self._computer_tool: ComputerTool | None = None
        self._chat_ctx: llm.ChatContext | None = None

        self._agent_loop_task: asyncio.Task[None] | None = None
        self._pending_messages: asyncio.Queue[str] = asyncio.Queue()
        self._started = False

    @property
    def page(self) -> BrowserPage | None:
        return self._page

    @property
    def session(self) -> BrowserSession | None:
        return self._session

    @property
    def computer_tool(self) -> ComputerTool | None:
        return self._computer_tool

    @property
    def chat_ctx(self) -> llm.ChatContext | None:
        return self._chat_ctx

    async def start(self, *, room: rtc.Room) -> None:
        """Initialize browser, session, and start listening for chat messages."""
        if self._started:
            return
        self._started = True
        self._room = room

        # 1. Create browser context and page
        self._browser_ctx = BrowserContext(dev_mode=False)
        await self._browser_ctx.initialize()

        self._page = await self._browser_ctx.new_page(
            url=self._url,
            width=self._width,
            height=self._height,
            framerate=self._framerate,
        )

        # 2. Create session and publish tracks
        self._session = BrowserSession(page=self._page, room=room)
        await self._session.start()

        # 3. Create computer tool
        from livekit.plugins.anthropic.computer_tool import ComputerTool

        self._computer_tool = ComputerTool(
            page=self._page,
            width=self._width,
            height=self._height,
        )

        # 4. Initialize chat context
        self._chat_ctx = llm.ChatContext()
        self._chat_ctx.add_message(role="system", content=self._instructions)

        # 5. Grant agent focus
        self._session.set_agent_focus(True)
        await self._page.send_focus_event(True)

        # 6. Listen for chat messages on data channel
        if self._chat_enabled:

            @room.on("data_received")
            def _on_chat_data(packet: rtc.DataPacket) -> None:
                if packet.topic != _CHAT_TOPIC:
                    return
                try:
                    data = json.loads(packet.data)
                    text = data.get("text", "")
                    if text:
                        self._pending_messages.put_nowait(text)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

            self._on_chat_data = _on_chat_data

        # 7. Start the agent loop
        self._agent_loop_task = asyncio.create_task(self._agent_loop())

    async def send_message(self, text: str) -> None:
        """Programmatically send a user message to the agent."""
        self._pending_messages.put_nowait(text)

    async def _agent_loop(self) -> None:
        """Main loop: wait for messages, call LLM, execute tools, repeat."""
        assert self._chat_ctx is not None
        assert self._computer_tool is not None
        assert self._session is not None

        while True:
            try:
                text = await self._pending_messages.get()

                if self._session.agent_interrupted.is_set():
                    self._session.agent_interrupted.clear()
                    self._session.set_agent_focus(True)
                    await self._page.send_focus_event(True)  # type: ignore[union-attr]

                self._chat_ctx.add_message(role="user", content=text)

                await self._send_status("thinking")

                await self._run_llm_loop()

                await self._send_status("idle")

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("error in agent loop")
                await self._send_status("idle")

    async def _run_llm_loop(self) -> None:
        """Call LLM, execute tool calls, loop until no more tool calls."""
        assert self._chat_ctx is not None
        assert self._computer_tool is not None
        assert self._session is not None

        all_tools: list[llm.Tool] = [*self._computer_tool.tools, *self._extra_tools]
        tool_ctx = llm.ToolContext(all_tools)

        while True:
            if self._session.agent_interrupted.is_set():
                logger.info("agent interrupted by human, pausing")
                await self._send_chat("(paused — you have control)")
                return

            response = await self._llm.chat(
                chat_ctx=self._chat_ctx,
                tools=all_tools,
            ).collect()

            if response.text:
                self._chat_ctx.add_message(role="assistant", content=response.text)
                await self._send_chat(response.text)

            if not response.tool_calls:
                return

            for tc in response.tool_calls:
                if self._session.agent_interrupted.is_set():
                    logger.info("agent interrupted between tool calls")
                    await self._send_chat("(paused — you have control)")
                    return

                if tc.name == "computer":
                    import json as _json

                    args = _json.loads(tc.arguments or "{}")
                    action = args.pop("action", "screenshot")

                    await self._send_status("acting")

                    screenshot_content = await self._computer_tool.execute(action, **args)

                    fnc_call = llm.FunctionCall(
                        call_id=tc.call_id,
                        name=tc.name,
                        arguments=tc.arguments or "{}",
                    )
                    fnc_output = llm.FunctionCallOutput(
                        call_id=tc.call_id,
                        name=tc.name,
                        output=json.dumps(screenshot_content),
                        is_error=False,
                    )
                    self._chat_ctx.items.append(fnc_call)
                    self._chat_ctx.items.append(fnc_output)
                else:
                    result = await llm.execute_function_call(tc, tool_ctx)
                    self._chat_ctx.items.append(result.fnc_call)
                    if result.fnc_call_out:
                        self._chat_ctx.items.append(result.fnc_call_out)

            await self._send_status("thinking")

    async def _send_chat(self, text: str) -> None:
        """Send a chat message to the frontend via data channel."""
        if self._room is None:
            return
        payload = json.dumps({"text": text, "sender": "agent"}).encode()
        try:
            await self._room.local_participant.publish_data(
                payload, reliable=True, topic=_CHAT_TOPIC
            )
        except Exception:
            logger.debug("failed to send chat message")

    async def _send_status(self, status: str) -> None:
        """Send agent status to the frontend via data channel."""
        if self._room is None:
            return
        payload = json.dumps({"status": status}).encode()
        try:
            await self._room.local_participant.publish_data(
                payload, reliable=True, topic=_STATUS_TOPIC
            )
        except Exception:
            logger.debug("failed to send status")

    async def aclose(self) -> None:
        """Clean up everything."""
        if self._agent_loop_task:
            self._agent_loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._agent_loop_task

        if self._computer_tool:
            self._computer_tool.aclose()

        if self._session:
            await self._session.aclose()

        if self._page:
            await self._page.aclose()

        if self._browser_ctx:
            await self._browser_ctx.aclose()

        if self._room and hasattr(self, "_on_chat_data"):
            self._room.off("data_received", self._on_chat_data)
