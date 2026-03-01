from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any

from livekit import rtc
from livekit.agents import llm
from livekit.agents.llm import function_tool
from livekit.browser import BrowserContext, BrowserPage  # type: ignore[import-untyped]

from .page_actions import PageActions
from .session import BrowserSession

if TYPE_CHECKING:
    from livekit.plugins.anthropic.computer_tool import ComputerTool

logger = logging.getLogger(__name__)

_POST_ACTION_DELAY = 0.8

_CHAT_TOPIC = "browser-agent-chat"
_STATUS_TOPIC = "browser-agent-status"
_CURSOR_TOPIC = "browser-agent-cursor"


class BrowserAgent:
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
        if self._started:
            return
        self._started = True
        self._room = room

        self._browser_ctx = BrowserContext(dev_mode=False)
        await self._browser_ctx.initialize()

        self._page = await self._browser_ctx.new_page(
            url=self._url,
            width=self._width,
            height=self._height,
            framerate=self._framerate,
        )

        self._session = BrowserSession(page=self._page, room=room)
        await self._session.start()

        from livekit.plugins.anthropic.computer_tool import ComputerTool

        self._page_actions = PageActions(page=self._page)
        self._computer_tool = ComputerTool(
            actions=self._page_actions,
            width=self._width,
            height=self._height,
        )

        @function_tool(name="navigate", description="Navigate the browser to a URL.")
        async def _navigate(url: str) -> None:
            pass

        @function_tool(name="go_back", description="Go back to the previous page.")
        async def _go_back() -> None:
            pass

        @function_tool(name="go_forward", description="Go forward to the next page.")
        async def _go_forward() -> None:
            pass

        self._nav_tools: list[llm.Tool] = [_navigate, _go_back, _go_forward]

        self._chat_ctx = llm.ChatContext()
        self._chat_ctx.add_message(role="system", content=self._instructions)

        await self._session.reclaim_agent_focus()

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

        self._agent_loop_task = asyncio.create_task(self._agent_loop())

    async def send_message(self, text: str) -> None:
        self._pending_messages.put_nowait(text)

    async def _agent_loop(self) -> None:
        assert self._chat_ctx is not None
        assert self._computer_tool is not None
        assert self._session is not None

        while True:
            try:
                text = await self._pending_messages.get()

                if self._session.agent_interrupted.is_set():
                    self._session.agent_interrupted.clear()
                    await self._session.reclaim_agent_focus()

                self._chat_ctx.add_message(role="user", content=text)

                await self._send_status("thinking")

                await self._run_llm_loop()

                await self._send_cursor_hide()
                await self._send_status("idle")

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("error in agent loop")
                await self._send_cursor_hide()
                await self._send_status("idle")

    async def _run_llm_loop(self) -> None:
        assert self._chat_ctx is not None
        assert self._computer_tool is not None
        assert self._session is not None

        all_tools: list[llm.Tool] = [
            *self._computer_tool.tools,
            *self._nav_tools,
            *self._extra_tools,
        ]
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

                    # Broadcast cursor position for frontend overlay
                    coord = args.get("coordinate")
                    if coord and len(coord) == 2:
                        await self._send_cursor_position(int(coord[0]), int(coord[1]), action)

                    await self._send_status("acting")

                    screenshot_content = await self._computer_tool.execute(action, **args)

                    # Wait for page to settle after clicks/typing
                    if action in (
                        "left_click",
                        "middle_click",
                        "key",
                        "type",
                    ):
                        await asyncio.sleep(_POST_ACTION_DELAY)

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
                elif tc.name in ("navigate", "go_back", "go_forward"):
                    await self._send_status("acting")
                    if tc.name == "navigate":
                        import json as _json

                        url = _json.loads(tc.arguments or "{}").get("url", "")
                        await self._page_actions.navigate(url)
                    elif tc.name == "go_back":
                        await self._page_actions.go_back()
                    else:
                        await self._page_actions.go_forward()
                    await asyncio.sleep(_POST_ACTION_DELAY)

                    screenshot_content = _screenshot_content(self._page_actions)
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
        if self._room is None:
            return
        payload = json.dumps({"status": status}).encode()
        try:
            await self._room.local_participant.publish_data(
                payload, reliable=True, topic=_STATUS_TOPIC
            )
        except Exception:
            logger.debug("failed to send status")

    async def _send_cursor_position(self, x: int, y: int, action: str) -> None:
        if self._room is None:
            return
        payload = json.dumps(
            {
                "x": x,
                "y": y,
                "action": action,
                "visible": True,
                "width": self._width,
                "height": self._height,
            }
        ).encode()
        try:
            await self._room.local_participant.publish_data(
                payload, reliable=True, topic=_CURSOR_TOPIC
            )
        except Exception:
            logger.debug("failed to send cursor position")

    async def _send_cursor_hide(self) -> None:
        if self._room is None:
            return
        payload = json.dumps({"visible": False}).encode()
        try:
            await self._room.local_participant.publish_data(
                payload, reliable=True, topic=_CURSOR_TOPIC
            )
        except Exception:
            pass

    async def aclose(self) -> None:
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


def _screenshot_content(actions: PageActions) -> list[dict[str, Any]]:
    from livekit.agents.utils.images import EncodeOptions, encode

    frame = actions.last_frame
    if frame is None:
        return [{"type": "text", "text": "(no frame available yet)"}]
    png_bytes = encode(frame, EncodeOptions(format="PNG"))
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64,
            },
        }
    ]
