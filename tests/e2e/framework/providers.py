"""Provider modes for an unmodified application.

`start_app(..., providers=Providers(script, config))` runs an ordinary application, such as an
example, with its LiveKit Inference models replaced before the application is imported:
`inference.LLM` and `llm="<model>"` strings become a `ScriptedLLM`, `inference.STT` and
`inference.TTS` the deterministic tone providers. The application's agents, tools, routing,
storage and transports stay real; a scripted model's output goes through the normal parser.

`script` names an importable module whose `reply(request)` answers each model request from
its content (the offered tools tell the roles apart) and raises `Unexpected` for anything it
was not written for. `config` is JSON, the same in every process of the app. With
`live_models=True` only speech is replaced: the application's own models run against LiveKit
Inference.

Job processes start by spawn or forkserver, so the replacement is installed by
`provider_site/sitecustomize.py` at every interpreter's startup, never by pickling.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from livekit.agents import llm

ENV = "E2E_PROVIDERS"
SITE = Path(__file__).with_name("provider_site")


@dataclass(frozen=True)
class Providers:
    script: str
    config: Mapping[str, Any] = field(default_factory=dict)
    live_models: bool = False

    def environment(self) -> dict[str, str]:
        return {
            ENV: json.dumps(
                {"script": self.script, "config": self.config, "live_models": self.live_models}
            )
        }


class Unexpected(AssertionError):
    """A model request the script was not written for."""


@dataclass(frozen=True)
class Request:
    """One model request: its chat context, the names of the tools it offers, and the config."""

    chat_ctx: llm.ChatContext
    tools: Sequence[str]
    config: Mapping[str, Any]


class Script(Protocol):
    def reply(self, request: Request) -> Any: ...


def journal(entry: Mapping[str, object]) -> None:
    """Append one record to the app's `models.jsonl`, the test's view of every model request."""
    directory = Path(os.environ["E2E_ARTIFACT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "models.jsonl").open("a") as log:
        log.write(json.dumps(entry, default=repr) + "\n")


def _items(chat_ctx: llm.ChatContext) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for item in chat_ctx.items:
        if item.type == "message":
            items.append({"type": "message", "role": item.role, "text": item.text_content})
        elif item.type == "function_call":
            items.append(
                {
                    "type": "call",
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": json.loads(item.arguments or "{}"),
                }
            )
        elif item.type == "function_call_output":
            items.append({"type": "output", "call_id": item.call_id, "output": item.output})
        else:
            items.append({"type": item.type})
    return items


def _tool_name(tool: object) -> str:
    name = getattr(tool, "name", None) or getattr(getattr(tool, "info", None), "name", None)
    return str(name)


def install() -> None:
    """Replace the providers named by `E2E_PROVIDERS`; a no-op when it is unset."""
    if ENV not in os.environ:
        return
    mode = json.loads(os.environ[ENV])
    config: Mapping[str, Any] = mode["config"]

    from livekit.agents import inference, voice
    from tests.fakes import FakeLLM

    from .media import ToneSTT, ToneTTS

    phrases = {int(pitch): text for pitch, text in config.get("phrases", {}).items()}
    inference.STT = lambda *args, **kwargs: ToneSTT(phrases)  # type: ignore[assignment,misc]
    inference.TTS = lambda *args, **kwargs: ToneTTS()  # type: ignore[assignment,misc]
    if mode["live_models"]:
        return

    script: Script = importlib.import_module(mode["script"])

    class ScriptedLLM(FakeLLM):
        def __init__(self, model: str) -> None:
            super().__init__(model=f"scripted:{model}")

        def _next(self, chat_ctx: llm.ChatContext, tools: list[llm.Tool]) -> Any:
            self.requests.append(chat_ctx.copy())
            names = [_tool_name(tool) for tool in tools]
            record: dict[str, object] = {"tools": names, "items": _items(chat_ctx)}
            try:
                reply = script.reply(Request(chat_ctx, names, config))
            except Exception as error:
                journal({**record, "unexpected": f"{type(error).__name__}: {error}"})
                raise
            parts = reply if isinstance(reply, list) else [reply]
            journal(
                {
                    **record,
                    "reply": [
                        {"call": part.name, "arguments": json.loads(part.arguments)}
                        if isinstance(part, llm.FunctionToolCall)
                        else repr(part)
                        if not isinstance(part, str)
                        else part
                        for part in parts
                    ],
                }
            )
            return reply

    def scripted(*args: object, model: str = "", **kwargs: object) -> ScriptedLLM:
        return ScriptedLLM(str(model or (args[0] if args else "")))

    inference.LLM = scripted  # type: ignore[assignment,misc]
    voice.llm_from_model_string = scripted  # type: ignore[attr-defined]
