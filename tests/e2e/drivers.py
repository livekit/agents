"""A2A actions and wire observations, using the upstream protocol records."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import httpx
from a2a.types import a2a_pb2 as pb
from google.protobuf import json_format
from google.protobuf.message import Message as ProtoMessage
from jsonrpc.jsonrpc2 import JSONRPC20Request, JSONRPC20Response  # type: ignore[import-untyped]
from pydantic import JsonValue, TypeAdapter

from livekit.agents.a2a.extension import EXTENSION_URI, to_json

from .resources import REQUEST_SECONDS, Child


class A2ACaller:
    def __init__(self, url: str, token: str) -> None:
        self.http = httpx.AsyncClient(
            base_url=url,
            headers={"Authorization": f"Bearer {token}", "A2A-Extensions": EXTENSION_URI},
            timeout=REQUEST_SECONDS,
            trust_env=False,
        )

    async def close(self) -> None:
        await self.http.aclose()

    def message(self, text: str, thread: str, *, channel: str = "") -> pb.Message:
        message = pb.Message(
            message_id=uuid.uuid4().hex,
            context_id=thread,
            role=pb.Role.ROLE_USER,
            parts=[pb.Part(text=text)],
        )
        if channel:
            message.metadata.update({f"{EXTENSION_URI}/channel": channel})
        return message

    async def send(self, message: pb.Message, *, endpoint: str = "desk") -> pb.Task:
        result = await self.rpc("message/send", pb.SendMessageRequest(message=message), endpoint)
        return json_format.ParseDict(result, pb.SendMessageResponse()).task

    async def rpc(
        self, method: str, params: ProtoMessage, endpoint: str = "desk", *, thread: str = ""
    ) -> dict[str, JsonValue]:
        wire_params = to_json(params)
        if thread:
            wire_params["contextId"] = thread
        async with asyncio.timeout(REQUEST_SECONDS):
            response = await self.http.post(
                f"/{endpoint}/v1",
                json=JSONRPC20Request(method=method, params=wire_params, _id=uuid.uuid4().hex).data,
            )
        response.raise_for_status()
        envelope = JSONRPC20Response.from_json(response.text)
        if envelope.error is not None:
            raise RuntimeError(f"{method}: {envelope.error}")
        return TypeAdapter(dict[str, JsonValue]).validate_python(envelope.result)

    async def stream(
        self, method: str, params: ProtoMessage, *, endpoint: str = "desk", thread: str = ""
    ) -> list[pb.StreamResponse]:
        return [
            frame async for frame in self.events(method, params, endpoint=endpoint, thread=thread)
        ]

    async def events(
        self, method: str, params: ProtoMessage, *, endpoint: str = "desk", thread: str = ""
    ) -> AsyncGenerator[pb.StreamResponse, None]:
        wire_params = to_json(params)
        if thread:
            # The current v0 server requires this extension to task lookup parameters.
            wire_params["contextId"] = thread
        async with (
            asyncio.timeout(REQUEST_SECONDS),
            self.http.stream(
                "POST",
                f"/{endpoint}/v1",
                json=JSONRPC20Request(method=method, params=wire_params, _id=uuid.uuid4().hex).data,
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    envelope = JSONRPC20Response.from_json(line[5:].strip())
                    if envelope.error is not None:
                        raise RuntimeError(f"{method}: {envelope.error}")
                    yield json_format.ParseDict(envelope.result, pb.StreamResponse())


def answer(task: pb.Task) -> str:
    return "".join(part.text for artifact in task.artifacts for part in artifact.parts)


@dataclass
class Workers:
    a: A2ACaller
    b: A2ACaller
    first: Child
    second: Child
