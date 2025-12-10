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
    Stream implementation for Google ADK responses.

    Handles Server-Sent Events (SSE) streaming from Google ADK server.
    """

    def __init__(
        self,
        *,
        llm_instance: "llm.LLM",
        api_base_url: str,
        app_name: str,
        user_id: str,
        session_id: str,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool] | None = None,
        conn_options: APIConnectOptions,
        **kwargs: Any,
    ) -> None:
        """
        Initialize ADK LLM Stream.

        Args:
            llm_instance: The parent LLM instance
            api_base_url: Base URL of ADK server
            app_name: ADK application name
            user_id: User identifier
            session_id: Session identifier
            chat_ctx: Chat context with messages
            tools: Optional function tools
            conn_options: API connection options
            **kwargs: Additional arguments
        """
        super().__init__(
            llm=llm_instance,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )
        self._api_base_url = api_base_url.rstrip("/")
        self._app_name = app_name
        self._user_id = user_id
        self._session_id = session_id
        self._client_session: aiohttp.ClientSession | None = None

    async def _run(self) -> None:
        """
        Execute the streaming request to ADK server.

        This method is called by the base LLMStream class.
        """
        logger.info(f"[ADK] Starting stream for session: {self._session_id}")

        # Create session if needed (handles pending session IDs)
        if self._session_id.startswith("__pending__"):
            logger.info("[ADK] Creating new ADK session from stream")
            # Create session inline to avoid event loop issues
            session_id = f"session-{int(time.time() * 1000)}"
            url = f"{self._api_base_url}/apps/{self._app_name}/users/{self._user_id}/sessions/{session_id}"

            # Use local client session for this request
            async with aiohttp.ClientSession() as temp_session:
                async with temp_session.post(url, json={}) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(f"Failed to create ADK session: {error_text}")
                    await resp.json()
                    self._session_id = session_id
                    logger.info(f"ADK session created: {session_id}")
        else:
            # For explicit session IDs (room/participant-based), ensure session exists
            logger.info(f"[ADK] Ensuring session exists: {self._session_id}")
            url = f"{self._api_base_url}/apps/{self._app_name}/users/{self._user_id}/sessions/{self._session_id}"

            async with aiohttp.ClientSession() as temp_session:
                async with temp_session.post(url, json={}) as resp:
                    if resp.status == 200:
                        await resp.json()
                        logger.info(f"[ADK] Session created or already exists: {self._session_id}")
                    elif resp.status == 409:
                        # Session already exists (Conflict)
                        logger.info(f"[ADK] Session already exists: {self._session_id}")
                    else:
                        # Log warning but continue (session might be managed externally)
                        error_text = await resp.text()
                        logger.warning(f"[ADK] Could not create session (status {resp.status}): {error_text}")

        # Get the last user message from chat context
        # ChatContext uses .items, not .messages
        last_item = self._chat_ctx.items[-1] if self._chat_ctx.items else None

        # Ensure it's a message (not function call, etc.) with USER role
        if last_item is None or last_item.type != "message":
            logger.warning("No message found in chat context")
            return

        last_message = last_item
        if last_message.role != "user":  # ChatRole is string literal
            logger.warning("No user message found in chat context")
            return

        # Extract text content from the message
        # LiveKit messages have a text_content property
        if hasattr(last_message, "text_content"):
            text_content = last_message.text_content
        elif isinstance(last_message.content, str):
            text_content = last_message.content
        elif isinstance(last_message.content, list):
            # Handle multi-part content (text + images, etc.)
            text_content = ""
            for part in last_message.content:
                if isinstance(part, str):
                    text_content += part
                elif hasattr(part, "text"):
                    text_content += part.text
        else:
            text_content = ""

        if not text_content:
            logger.warning("Empty text content in user message")
            return

        # Prepare ADK request payload
        payload = {
            "app_name": self._app_name,
            "user_id": self._user_id,
            "session_id": self._session_id,
            "new_message": {
                "role": "user",
                "parts": [{"text": text_content}]
            },
            "streaming": True,
        }

        # Add tools if provided
        if self._tools:
            payload["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in self._tools
            ]

        logger.info(f"[ADK] Sending message to ADK: {text_content[:100]}")
        logger.debug(f"[ADK] Full payload: {payload}")

        # Create client session
        self._client_session = aiohttp.ClientSession()
        try:
            logger.debug(f"[ADK] POST {self._api_base_url}/run_sse")
            async with self._client_session.post(
                f"{self._api_base_url}/run_sse",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"[ADK] Request failed: {resp.status} - {error_text}")
                    raise RuntimeError(
                        f"ADK request failed (status {resp.status}): {error_text}"
                    )

                logger.info(f"[ADK] Streaming response started (status {resp.status})")

                # Parse SSE stream
                request_id = f"{self._session_id}-{int(time.time() * 1000)}"

                async for line_bytes in resp.content:
                    line = line_bytes.decode("utf-8").strip()

                    if not line:
                        continue

                    # Skip SSE comments
                    if line.startswith(":"):
                        continue

                    # Parse data lines
                    if line.startswith("data:"):
                        data_str = line[len("data:"):].strip()
                        if not data_str:
                            continue

                        try:
                            event = json.loads(data_str)

                            # Extract content from ADK response
                            content = event.get("content", {})
                            parts = content.get("parts", [])
                            is_partial = event.get("partial", False)

                            if parts and is_partial:
                                text = parts[0].get("text", "")
                                if text:
                                    logger.info(f"[ADK] Chunk: {text}")

                                    # Create and push ChatChunk
                                    # ChatChunk takes delta directly, not choices list
                                    chunk = llm.ChatChunk(
                                        id=request_id,
                                        delta=llm.ChoiceDelta(
                                            role="assistant",  # ChatRole is string literal
                                            content=text,
                                        ),
                                    )
                                    self._event_ch.send_nowait(chunk)

                            # Handle tool calls if present
                            if "tool_calls" in event:
                                # TODO: Implement tool calling support
                                logger.info(f"Tool calls detected: {event['tool_calls']}")

                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse SSE data: {e}")
                            continue

        except aiohttp.ClientError as e:
            logger.error(f"[ADK] Network error: {e}", exc_info=True)
            raise RuntimeError(f"Failed to communicate with ADK server: {e}") from e
        except Exception as e:
            logger.error(f"[ADK] Unexpected error: {e}", exc_info=True)
            raise
        finally:
            if self._client_session and not self._client_session.closed:
                await self._client_session.close()
                self._client_session = None
