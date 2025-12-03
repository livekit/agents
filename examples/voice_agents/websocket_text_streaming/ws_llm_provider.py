"""
Custom WebSocket LLM Provider for LiveKit Agents

This module provides a custom LLM implementation that connects to a WebSocket server
instead of standard LLM providers like OpenAI. It enables LiveKit Agents to use
any WebSocket-based LLM backend.

Usage:
    from ws_llm_provider import WebSocketLLM
    
    llm = WebSocketLLM(ws_url="ws://localhost:8765")
    
    # Use with AgentSession
    session = AgentSession(llm=llm, ...)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from websockets.asyncio.client import connect

from livekit.agents import llm
from livekit.agents.llm import ChatChunk, ChatContext, ChoiceDelta
from livekit.agents.llm.tool_context import FunctionTool, RawFunctionTool
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)


@dataclass
class WebSocketLLMOptions:
    """Configuration options for WebSocket LLM."""
    ws_url: str
    model_name: str


class WebSocketLLM(llm.LLM):
    """
    Custom LLM implementation that connects to a WebSocket server.
    
    This allows LiveKit Agents to use any WebSocket-based LLM backend
    instead of standard providers like OpenAI.
    
    Args:
        ws_url: WebSocket server URL (e.g., "ws://localhost:8765")
        model_name: Optional model name identifier for metrics/logging
    """

    def __init__(
        self,
        *,
        ws_url: str = "ws://localhost:8765",
        model_name: str = "websocket-llm",
    ) -> None:
        super().__init__()
        self._opts = WebSocketLLMOptions(
            ws_url=ws_url,
            model_name=model_name,
        )

    @property
    def model(self) -> str:
        """Get the model name for this LLM instance."""
        return self._opts.model_name

    @property
    def provider(self) -> str:
        """Get the provider name for this LLM instance."""
        return "websocket"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> llm.LLMStream:
        """
        Create a new chat completion stream.
        
        Args:
            chat_ctx: The chat context containing conversation history
            tools: Not supported in this PoC
            conn_options: Connection options for retries
            parallel_tool_calls: Not supported in this PoC
            tool_choice: Not supported in this PoC
            extra_kwargs: Additional arguments (ignored)
            
        Returns:
            WebSocketLLMStream instance for streaming responses
        """
        return WebSocketLLMStream(
            llm=self,
            ws_url=self._opts.ws_url,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


class WebSocketLLMStream(llm.LLMStream):
    """
    Streaming implementation that connects to WebSocket LLM server.
    
    This class handles the actual communication with the WebSocket server
    and converts responses into ChatChunk events.
    """

    def __init__(
        self,
        llm: WebSocketLLM,
        *,
        ws_url: str,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._ws_url = ws_url
        self._request_id = str(uuid.uuid4())

    async def _run(self) -> None:
        """
        Execute the LLM request via WebSocket.
        
        This method:
        1. Converts ChatContext to WebSocket message format
        2. Sends request to WebSocket server
        3. Receives streaming responses
        4. Emits ChatChunk events for each token
        """
        # Convert chat context to simple message format
        messages = self._chat_ctx_to_messages(self._chat_ctx)

        # Connect to WebSocket server and stream response
        async with connect(self._ws_url) as websocket:
            # Send chat request
            request = {
                "type": "chat",
                "messages": messages,
            }
            await websocket.send(json.dumps(request))

            # Receive streaming response
            first_chunk = True
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "delta":
                    # Emit delta chunk
                    content = data.get("content", "")
                    chunk = ChatChunk(
                        id=self._request_id,
                        delta=ChoiceDelta(
                            role="assistant" if first_chunk else None,
                            content=content,
                        ),
                    )
                    self._event_ch.send_nowait(chunk)
                    first_chunk = False

                elif msg_type == "complete":
                    # Stream is complete, break the loop
                    break

                elif msg_type == "error":
                    # Handle error from server
                    error_msg = data.get("message", "Unknown error")
                    raise Exception(f"WebSocket LLM error: {error_msg}")

    def _chat_ctx_to_messages(self, chat_ctx: ChatContext) -> list[dict[str, str]]:
        """
        Convert ChatContext to simple message format for WebSocket protocol.
        
        Args:
            chat_ctx: LiveKit ChatContext with conversation history
            
        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        messages: list[dict[str, str]] = []

        for item in chat_ctx.items:
            if item.type == "message":
                # Extract text content from the message
                text_content = item.text_content
                if text_content:
                    messages.append({
                        "role": item.role,
                        "content": text_content,
                    })

        return messages

