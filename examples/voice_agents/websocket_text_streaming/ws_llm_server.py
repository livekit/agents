"""
Standalone WebSocket LLM Server - Healthcare Assistant

This is a simple WebSocket server that acts as a healthcare-focused LLM assistant.
It uses OpenAI as the backend but streams responses over WebSocket.
This server is completely independent of LiveKit.

Protocol:
    Client -> Server: {"type": "chat", "messages": [{"role": "user", "content": "..."}]}
    Server -> Client: {"type": "delta", "content": "..."} (multiple times during streaming)
    Server -> Client: {"type": "complete", "content": "full response"}
    Server -> Client: {"type": "error", "message": "error description"}

Usage:
    uv run ws_llm_server.py

    The server will start on ws://localhost:8765
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import openai
from dotenv import load_dotenv
from websockets.asyncio.server import ServerConnection, serve

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ws-llm-server")

# Healthcare-focused system prompt
HEALTHCARE_SYSTEM_PROMPT = """You are a helpful, empathetic healthcare assistant. Your role is to:

1. Provide general health information and wellness guidance
2. Help users understand medical terminology in simple terms
3. Encourage users to seek professional medical advice for specific conditions
4. Offer emotional support and active listening
5. Provide information about healthy lifestyle choices

Important guidelines:
- Always recommend consulting with a healthcare professional for specific medical advice
- Never diagnose conditions or prescribe treatments
- Be empathetic and understanding
- Use clear, simple language
- If someone describes an emergency, advise them to call emergency services immediately

Remember: You are not a replacement for professional medical care. Your role is to inform, 
support, and guide users toward appropriate healthcare resources."""


class HealthcareLLMServer:
    """WebSocket server that provides healthcare LLM assistance."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        openai_model: str = "gpt-4.1",
    ):
        self.host = host
        self.port = port
        self.openai_model = openai_model
        self._client = openai.AsyncOpenAI()

    async def handle_connection(self, websocket: ServerConnection) -> None:
        """Handle a single WebSocket connection."""
        client_id = id(websocket)
        logger.info(f"Client {client_id} connected")

        # Maintain conversation history for this connection
        conversation_history: list[dict[str, Any]] = [
            {"role": "system", "content": HEALTHCARE_SYSTEM_PROMPT}
        ]

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(websocket, data, conversation_history)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON format")
                except Exception as e:
                    logger.exception(f"Error handling message from client {client_id}")
                    await self._send_error(websocket, str(e))
        except Exception as e:
            logger.info(f"Client {client_id} disconnected: {e}")
        finally:
            logger.info(f"Client {client_id} connection closed")

    async def _handle_message(
        self,
        websocket: ServerConnection,
        data: dict[str, Any],
        conversation_history: list[dict[str, Any]],
    ) -> None:
        """Process incoming message and generate streaming response."""
        msg_type = data.get("type")

        if msg_type == "chat":
            messages = data.get("messages", [])
            if not messages:
                await self._send_error(websocket, "No messages provided")
                return

            # Add new messages to conversation history
            for msg in messages:
                if msg.get("role") and msg.get("content"):
                    conversation_history.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            # Generate streaming response
            await self._generate_streaming_response(websocket, conversation_history)

        elif msg_type == "reset":
            # Reset conversation history (keep system prompt)
            conversation_history.clear()
            conversation_history.append({
                "role": "system",
                "content": HEALTHCARE_SYSTEM_PROMPT
            })
            await websocket.send(json.dumps({
                "type": "reset_complete",
                "message": "Conversation history cleared"
            }))

        else:
            await self._send_error(websocket, f"Unknown message type: {msg_type}")

    async def _generate_streaming_response(
        self,
        websocket: ServerConnection,
        conversation_history: list[dict[str, Any]],
    ) -> None:
        """Generate and stream LLM response."""
        full_response = ""

        try:
            stream = await self._client.chat.completions.create(
                model=self.openai_model,
                messages=conversation_history,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content

                    # Send delta to client
                    await websocket.send(json.dumps({
                        "type": "delta",
                        "content": content
                    }))

            # Send complete message
            await websocket.send(json.dumps({
                "type": "complete",
                "content": full_response
            }))

            # Add assistant response to conversation history
            conversation_history.append({
                "role": "assistant",
                "content": full_response
            })

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            await self._send_error(websocket, f"LLM API error: {e.message}")

    async def _send_error(self, websocket: ServerConnection, message: str) -> None:
        """Send error message to client."""
        await websocket.send(json.dumps({
            "type": "error",
            "message": message
        }))

    async def start(self) -> None:
        """Start the WebSocket server."""
        logger.info(f"Starting Healthcare LLM Server on ws://{self.host}:{self.port}")

        async with serve(self.handle_connection, self.host, self.port):
            logger.info("Server started. Press Ctrl+C to stop.")
            await asyncio.Future()  # Run forever


async def main():
    """Main entry point."""
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is not set")
        return

    server = HealthcareLLMServer(
        host="localhost",
        port=8765,
        openai_model="gpt-4o-mini",
    )
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())

