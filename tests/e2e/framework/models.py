"""Explicit live providers and reviewable conversation/communication artifacts."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message

from livekit.agents import SessionState, _core
from livekit.agents.inference import LLM


def live_model_environment() -> dict[str, str]:
    """Resolve before a room app replaces LIVEKIT_URL with its isolated SFU address."""
    key = os.environ.get("LIVEKIT_INFERENCE_API_KEY") or os.environ.get("LIVEKIT_API_KEY")
    secret = os.environ.get("LIVEKIT_INFERENCE_API_SECRET") or os.environ.get("LIVEKIT_API_SECRET")
    if not key or not secret:
        raise RuntimeError("live evaluations require configured LiveKit Inference credentials")
    return {
        "LIVEKIT_INFERENCE_API_KEY": key,
        "LIVEKIT_INFERENCE_API_SECRET": secret,
        "LIVEKIT_INFERENCE_URL": _core.inference_url(),
    }


@asynccontextmanager
async def live_models(*, frontend: str, routing: str) -> AsyncIterator[tuple[LLM, LLM]]:
    """Construct separate models in their owning process; never substitute scripted inference."""
    env = live_model_environment()
    async with AsyncExitStack() as resources:
        models = []
        for model_id in (frontend, routing):
            model = LLM(
                model_id,
                api_key=env["LIVEKIT_INFERENCE_API_KEY"],
                api_secret=env["LIVEKIT_INFERENCE_API_SECRET"],
                base_url=env["LIVEKIT_INFERENCE_URL"],
            )
            resources.push_async_callback(model.aclose)
            models.append(model)
        yield models[0], models[1]


def _payload(value: object) -> object:
    if isinstance(value, Message):
        return MessageToDict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return repr(value)


def write_live_report(
    directory: Path,
    *,
    models: Mapping[str, str],
    turns: Sequence[Mapping[str, object]],
    state: SessionState,
    observations: Mapping[str, object] | None = None,
) -> Path:
    """Keep perceived conversation separate from the SDK communication journal and task state."""
    snapshot = state.snapshot()
    report = {
        "models": dict(models),
        "turns": list(turns),
        "observations": dict(observations or {}),
        "agents": {
            agent_id: {"agent_name": agent.agent_name, "chat_ctx": agent.chat_ctx.to_dict()}
            for agent_id, agent in snapshot.agents.items()
        },
        "conversation": [
            {
                "id": message.id,
                "role": message.role,
                "text": message.text_content,
                "interrupted": message.interrupted,
            }
            for message in snapshot.chat_history.items
            if message.type == "message"
        ],
        "remote_agents": {
            remote_id: {
                "messages": {
                    message_id: _payload(message) for message_id, message in remote.messages.items()
                },
                "tasks": {
                    task_id: {
                        "task": _payload(task.task),
                        "input_requests": {
                            request_id: asdict(request)
                            for request_id, request in task.input_requests.items()
                        },
                    }
                    for task_id, task in remote.tasks.items()
                },
                "events": [
                    {
                        "sequence": event.sequence,
                        "request_id": event.request_id,
                        "kind": event.kind,
                        "payload": _payload(event.payload),
                    }
                    for event in remote.events
                ],
            }
            for remote_id, remote in snapshot.remote_agents.items()
        },
    }
    path = directory / "live-model-report.json"
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    pending.replace(path)
    return path
