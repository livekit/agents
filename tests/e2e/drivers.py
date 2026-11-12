"""A2A actions and wire observations, using the upstream protocol records."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass

import httpx
from a2a.client import Client, ClientConfig, ClientFactory
from a2a.client.card_resolver import A2ACardResolver
from a2a.types import a2a_pb2 as pb
from a2a.utils.constants import TransportProtocol
from google.protobuf import json_format
from google.protobuf.message import Message as ProtoMessage
from jsonrpc.jsonrpc2 import (  # type: ignore[import-untyped]  # json-rpc ships no types
    JSONRPC20Request,
    JSONRPC20Response,
)
from pydantic import JsonValue, TypeAdapter

from livekit.agents.a2a import EXTENSION_URI

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

    def message(self, text: str, thread: str) -> pb.Message:
        return pb.Message(
            message_id=uuid.uuid4().hex,
            context_id=thread,
            role=pb.Role.ROLE_USER,
            parts=[pb.Part(text=text)],
        )

    async def send(self, message: pb.Message, *, endpoint: str = "desk") -> pb.Task:
        result = await self.rpc("message/send", pb.SendMessageRequest(message=message), endpoint)
        return json_format.ParseDict(result, pb.SendMessageResponse()).task

    async def rpc(
        self, method: str, params: ProtoMessage, endpoint: str = "desk", *, thread: str = ""
    ) -> dict[str, JsonValue]:
        response = await self.http.post(f"/{endpoint}/v1", json=_request(method, params, thread))
        response.raise_for_status()
        envelope = JSONRPC20Response.from_json(response.text)
        if envelope.error is not None:
            raise RuntimeError(f"{method}: {envelope.error}")
        return TypeAdapter(dict[str, JsonValue]).validate_python(envelope.result)

    async def events(
        self, method: str, params: ProtoMessage, *, endpoint: str = "desk", thread: str = ""
    ) -> AsyncGenerator[pb.StreamResponse, None]:
        async with self.http.stream(
            "POST", f"/{endpoint}/v1", json=_request(method, params, thread)
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    envelope = JSONRPC20Response.from_json(line[5:].strip())
                    if envelope.error is not None:
                        raise RuntimeError(f"{method}: {envelope.error}")
                    yield json_format.ParseDict(envelope.result, pb.StreamResponse())


def _request(method: str, params: ProtoMessage, thread: str) -> dict[str, object]:
    wire_params = json_format.MessageToDict(params)
    if thread:
        # The current server requires the thread beside task lookup parameters.
        wire_params["contextId"] = thread
    request: dict[str, object] = JSONRPC20Request(
        method=method, params=wire_params, _id=uuid.uuid4().hex
    ).data
    return request


def answer(task: pb.Task) -> str:
    return "".join(part.text for artifact in task.artifacts for part in artifact.parts)


@dataclass
class Workers:
    a: A2ACaller
    b: A2ACaller
    a_process: Child
    b_process: Child


# -- a stock a2a-sdk client of an in-process server (``resources.serving``) --

ENDPOINT = "desk"


def endpoint_url(http: httpx.AsyncClient) -> str:
    return f"{str(http.base_url).rstrip('/')}/{ENDPOINT}"


async def stock_client(http: httpx.AsyncClient) -> Client:
    """A vanilla a2a-sdk client over JSON-RPC, without the extension header."""
    card = await A2ACardResolver(http, endpoint_url(http)).get_agent_card()
    return ClientFactory(
        ClientConfig(
            httpx_client=http,
            streaming=True,
            supported_protocol_bindings=[TransportProtocol.JSONRPC],
            accepted_output_modes=["text/plain"],
        )
    ).create(card)


async def send(client: Client, text: str, *, context_id: str = "") -> pb.Task:
    """Sends ``text`` (on ``context_id``'s thread, else a new one); the task its stream opens."""
    message = pb.Message(
        message_id=uuid.uuid4().hex,
        context_id=context_id,
        role=pb.Role.ROLE_USER,
        parts=[pb.Part(text=text)],
    )
    events = [event async for event in client.send_message(pb.SendMessageRequest(message=message))]
    return events[0].task


async def rpc(http: httpx.AsyncClient, method: str, params: Mapping[str, object]) -> httpx.Response:
    return await http.post(
        f"{endpoint_url(http)}/v1",
        json={"jsonrpc": "2.0", "id": method, "method": method, "params": dict(params)},
    )


async def first_event(http: httpx.AsyncClient, ctx_id: str) -> tuple[str, httpx.Response]:
    """A ``message/stream`` whose first event, the task, was read; the response stays open."""
    message = {
        "messageId": "m1",
        "role": "ROLE_USER",
        "parts": [{"text": "slow"}],
        "contextId": ctx_id,
    }
    request = http.build_request(
        "POST",
        f"{endpoint_url(http)}/v1",
        json={
            "jsonrpc": "2.0",
            "id": "initial",
            "method": "message/stream",
            "params": {"message": message},
        },
    )
    response = await http.send(request, stream=True)
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            return str(json.loads(line[len("data: ") :])["result"]["task"]["id"]), response
    raise AssertionError("the stream ended before its first event")


async def settled_task(http: httpx.AsyncClient, task_id: str, ctx_id: str) -> pb.Task:
    """The task as ``tasks/get`` returns it once it is no longer working."""
    while True:
        response = await rpc(http, "tasks/get", {"id": task_id, "contextId": ctx_id})
        task = json_format.ParseDict(response.json()["result"], pb.Task())
        if task.status.state != pb.TaskState.TASK_STATE_WORKING:
            return task
        await asyncio.sleep(0.05)
