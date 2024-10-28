# Copyright 2024 Riino.Site (https://riino.site)

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

from __future__ import annotations

import base64
import inspect
import json
import os
from dataclasses import dataclass
from typing import Any, Awaitable, List, Tuple, get_args, get_origin
import  asyncio
import httpx
from livekit import rtc
from livekit.agents import llm, utils
from .models import (
    ChatModels,
)
from .log import logger

def build_message(msg: llm.Message) -> dict:

    return {
        "role": msg.role,
        "content": msg.content
    }

@dataclass
class LLMOptions:
    model: str | ChatModels
    user: str | None
    temperature: float | None

class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "dify",#will not be used
        api_key: str | None = None,
        base_url: str |  None = "https://api.dify.ai/v1",
        user: str | None = None,
        client: httpx.AsyncClient | None = None,
        temperature: float | None = None,#will not be used
    ) -> None:
        """
        Create a new instance of Telnyx LLM.

        ``api_key`` must be set to your Dify App API key, either using the argument or by setting
        the ``DIFY_API_KEY`` environmental variable.
        """
        api_key = api_key or os.environ.get("DIFY_API_KEY")
        if api_key is None:
            raise ValueError("Please")
        self.base_url = base_url or "https://api.dify.ai/v1"
        self._opts = LLMOptions(
            model=model,
            user=user,
            temperature=temperature
        )
        
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            # timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
            follow_redirects=True,
            timeout=httpx.Timeout(
                connect=15.0,    # Connect Timeout
                read=300.0,      # 5-min Read Timeout
                write=30.0,      # Write Timeout
                pool=5.0
            ),
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=50,
                keepalive_expiry=120,
            ),
        )
        self._conversation_id = ""

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> "LLMStream":
        last_message = chat_ctx.messages[-1] if chat_ctx.messages else None
        
        request_data = {
            "inputs": {},
            "query": last_message.content if last_message else "",
            "response_mode": "streaming",#must be streaming
            "conversation_id": self._conversation_id,  
            "user": self._opts.user or "livekit-plugin-dify",
            #no temperature
        }

        stream = self._client.post(
            '/chat-messages',
            json=request_data,
        )

        return LLMStream(
            dify_stream=stream,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            conversation_id_callback=self._update_conversation_id  # pass callback to update conversation_id
        )

    def _update_conversation_id(self, new_id: str) -> None:
        """
        Callback for conversation id update
        """
        self._conversation_id = new_id

    async def close(self) -> None:
        """Close Connection"""
        if self._client:
            await self._client.aclose()

class LLMStream(llm.LLMStream):
    def __init__(
        self,
        *,
        dify_stream,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
        conversation_id_callback: callable,
    ) -> None:
        super().__init__(chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)
        self._awaitable_dify_stream = dify_stream
        self._dify_stream = None
        self._conversation_id_callback = conversation_id_callback
        self._conversation_id_updated = False
        self._current_count = 0
        self._skip_interval = 1  

    async def aclose(self) -> None:
        if self._dify_stream:
            await self._dify_stream.close()
        return await super().aclose()

    async def __anext__(self):
        if not self._dify_stream:
            # print("Initializing stream...")
            self._dify_stream = await self._awaitable_dify_stream
            # print("Stream initialized.")

        async for chunk in self._dify_stream.aiter_lines():

            if not chunk.strip():
                await asyncio.sleep(0.1)  
                continue

            # print(f"Received chunk: {chunk.strip()}")  

            self._current_count += 1
            # print(f"Current count: {self._current_count}, Skip interval: {self._skip_interval}")  # 跳过计数和间隔

            if self._current_count < self._skip_interval:
                # print("Skipping this chunk.")
                continue  
            else:
                # print("Processing this chunk.")
                self._current_count = 0  

            event_data = chunk[len("data: "):].strip()
            try:
                message = json.loads(event_data)
                # print(f"Parsed message: {message}")  
            except json.JSONDecodeError:
                # print("Failed to parse JSON, skipping this chunk.")
                logger.warning(
                "Failed to parse JSON, skipping this chunk."
                )
                continue

            if 'answer' in message:
                chat_chunk = self._parse_message(message)
                if chat_chunk is not None:
                    self._skip_interval += 1  
                    return chat_chunk
            else:
                pass
                # print("No 'answer' key found in message, skipping.")

        # print("No more chunks to process, stopping iteration.")
        raise StopAsyncIteration



    def _parse_message(self, message: dict) -> llm.ChatChunk | None:
        if not self._conversation_id_updated and "conversation_id" in message:
            self._conversation_id_callback(message["conversation_id"])
            self._conversation_id_updated = True

        if message.get("event") == "message":
                return llm.ChatChunk(
                    choices=[
                        llm.Choice(
                            delta=llm.ChoiceDelta(
                                content=message["answer"],
                                role="assistant"
                            ),
                            index=0
                        )
                    ]
                )
        else:
            return None