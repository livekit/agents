"""Google ADK LLM Stream implementation."""

import json
import logging
import time
from typing import Any

import aiohttp

from livekit.agents import llm
from livekit.agents.types import APIConnectOptions

logger = logging.getLogger(__name__)


class LLMStream(llm.LLMStream):
    """
    SSE streaming implementation for Google ADK.
    Text-only LLM (no audio).
    """

    def __init__(
        self,
        *,
        llm_instance: llm.LLM,
        api_base_url: str,
        app_name: str,
        user_id: str,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool] | None,
        conn_options: APIConnectOptions,
        **_: Any,
    ) -> None:
        super().__init__(
            llm=llm_instance,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )
        self._api_base_url = api_base_url.rstrip("/")
        self._app_name = app_name
        self._user_id = user_id
        self._emitted_len = 0
        self._cancelled = False

    async def aclose(self) -> None:
        self._cancelled = True
        await super().aclose()

    async def _run(self) -> None:
        # LLM owns lifecycle, but stream triggers it
        self._session_id = await self._llm._ensure_session()
        self._client = await self._llm._ensure_client_session()

        logger.info("[ADK] Stream start (session=%s)", self._session_id)

        last = self._chat_ctx.items[-1] if self._chat_ctx.items else None
        if not last or last.type != "message" or last.role != "user":
            self._event_ch.close()
            return

        text = getattr(last, "text_content", "") or ""
        if not text:
            self._event_ch.close()
            return

        # Tools should NOT be passed through LiveKit - they are configured in ADK
        if self._tools:
            raise ValueError(
                "Tools should not be passed through LiveKit when using Google ADK. "
                "Tools must be registered directly in your ADK application configuration. "
                "Remove the 'tools' parameter from your Agent() and configure tools in ADK instead."
            )

        payload = {
            "app_name": self._app_name,
            "user_id": self._user_id,
            "session_id": self._session_id,
            "new_message": {"role": "user", "parts": [{"text": text}]},
            "streaming": True,
        }

        request_id = f"{self._session_id}-{int(time.time() * 1000)}"

        try:
            async with self._client.post(
                f"{self._api_base_url}/run_sse",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(await resp.text())

                async for raw in resp.content:
                    if self._cancelled:
                        logger.info("[ADK] Stream cancelled")
                        self._event_ch.close()
                        return

                    line = raw.decode().strip()
                    if not line or line.startswith(":"):
                        continue

                    if not line.startswith("data:"):
                        continue

                    try:
                        event = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue

                    parts = event.get("content", {}).get("parts", [])
                    text_now = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
                    if not text_now:
                        continue

                    delta = text_now[self._emitted_len :]
                    self._emitted_len += len(delta)

                    if delta:
                        self._event_ch.send_nowait(
                            llm.ChatChunk(
                                id=request_id,
                                delta=llm.ChoiceDelta(role="assistant", content=delta),
                            )
                        )

                    if not event.get("partial", False):
                        self._event_ch.send_nowait(
                            llm.ChatChunk(
                                id=request_id,
                                delta=llm.ChoiceDelta(role="assistant", content=""),
                                finish_reason="stop",
                            )
                        )
                        self._event_ch.close()
                        return

                # Defensive termination
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id=request_id,
                        delta=llm.ChoiceDelta(role="assistant", content=""),
                        finish_reason="stop",
                    )
                )
                self._event_ch.close()

        except Exception:
            self._event_ch.close()
            raise
