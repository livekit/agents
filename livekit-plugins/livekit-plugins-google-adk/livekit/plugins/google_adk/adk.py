from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from livekit.agents import APIConnectionError, llm
from livekit.agents.llm import ToolChoice
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

if TYPE_CHECKING:
    from livekit.agents.llm.chat_context import ChatMessage

_DEFAULT_APP_NAME = "LiveKitGoogleADK"
_DEFAULT_USER_ID = "livekit-user"
_MISSING_ADK_MESSAGE = (
    "Google ADK support requires the optional 'google-adk' dependency. "
    "Install it with: pip install livekit-plugins-google-adk google-adk"
)


@dataclass(frozen=True)
class _ADKModules:
    InMemoryRunner: type[Any]
    Content: type[Any]
    Event: type[Any]
    Part: type[Any]


@lru_cache(maxsize=1)
def _adk_modules() -> _ADKModules:
    try:
        runners = importlib.import_module("google.adk.runners")
        events = importlib.import_module("google.adk.events.event")
        types_mod = importlib.import_module("google.genai.types")
    except ImportError as e:
        raise ImportError(_MISSING_ADK_MESSAGE) from e

    return _ADKModules(
        InMemoryRunner=runners.InMemoryRunner,
        Content=types_mod.Content,
        Event=events.Event,
        Part=types_mod.Part,
    )


class LLMAdapter(llm.LLM):
    def __init__(
        self,
        runner: Any | None = None,
        *,
        agent: Any | None = None,
        app: Any | None = None,
        app_name: str = _DEFAULT_APP_NAME,
        user_id: str = _DEFAULT_USER_ID,
        include_thoughts: bool = False,
        assistant_name: str | None = None,
        run_config: Any | None = None,
    ) -> None:
        super().__init__()
        mods = _adk_modules()

        if runner is None and agent is None and app is None:
            raise ValueError("Provide either runner=..., agent=..., or app=...")

        if runner is None:
            runner = mods.InMemoryRunner(agent=agent, app=app, app_name=app_name)
            self._owns_runner = True
        else:
            self._owns_runner = False

        self._mods = mods
        self._runner = runner
        self._user_id = user_id
        self._include_thoughts = include_thoughts
        self._assistant_name = assistant_name or getattr(getattr(runner, "agent", None), "name", None)
        self._run_config = run_config

    @property
    def model(self) -> str:
        model = getattr(getattr(self._runner, "agent", None), "model", None)
        if isinstance(model, str) and model:
            return model
        if model is None:
            return "unknown"
        return type(model).__name__

    @property
    def provider(self) -> str:
        return "Google ADK"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        # unused because tool execution is expected to happen in ADK itself
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> ADKStream:
        return ADKStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            runner=self._runner,
            user_id=self._user_id,
            include_thoughts=self._include_thoughts,
            assistant_name=self._assistant_name or "assistant",
            run_config=self._run_config,
            mods=self._mods,
        )

    async def aclose(self) -> None:
        if not self._owns_runner:
            return

        close = getattr(self._runner, "close", None)
        if close is None:
            return

        result = close()
        if inspect.isawaitable(result):
            await result


class ADKStream(llm.LLMStream):
    def __init__(
        self,
        llm_adapter: LLMAdapter,
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
        runner: Any,
        user_id: str,
        include_thoughts: bool,
        assistant_name: str,
        run_config: Any | None,
        mods: _ADKModules,
    ) -> None:
        super().__init__(
            llm_adapter,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )
        self._runner = runner
        self._user_id = user_id
        self._include_thoughts = include_thoughts
        self._assistant_name = assistant_name
        self._run_config = run_config
        self._mods = mods
        self._request_id = uuid4().hex

    async def _run(self) -> None:
        session_id = uuid4().hex
        session = await self._runner.session_service.create_session(
            app_name=self._runner.app_name,
            user_id=self._user_id,
            session_id=session_id,
        )

        try:
            prompt = await self._populate_session_and_build_prompt(session)
            content = self._mods.Content(
                role="user",
                parts=[_part_from_text(self._mods, prompt)],
            )

            emitted_text = ""
            async for event in self._runner.run_async(
                user_id=self._user_id,
                session_id=session_id,
                new_message=content,
                run_config=self._run_config,
            ):
                error_message = getattr(event, "error_message", None)
                if error_message:
                    raise APIConnectionError(
                        f"google adk error: {error_message}",
                        retryable=False,
                    )

                text = _event_text(event, include_thoughts=self._include_thoughts)
                if not text:
                    continue

                delta, emitted_text = _delta_text(text, emitted_text)
                if not delta:
                    continue

                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id=self._request_id,
                        delta=llm.ChoiceDelta(role="assistant", content=delta),
                    )
                )
        finally:
            delete_session = getattr(self._runner.session_service, "delete_session", None)
            if delete_session is not None:
                result = delete_session(
                    app_name=self._runner.app_name,
                    user_id=self._user_id,
                    session_id=session_id,
                )
                if inspect.isawaitable(result):
                    await result

    async def _populate_session_and_build_prompt(self, session: Any) -> str:
        instructions, conversation = _split_chat_ctx(self.chat_ctx)

        prompt_body = ""
        history = conversation
        if conversation and conversation[-1].role == "user":
            prompt_body = conversation[-1].text_content
            history = conversation[:-1]
        elif conversation:
            prompt_body = _render_transcript(conversation)
            history = []

        for message in history:
            event = _chat_message_to_adk_event(self._mods, message, self._assistant_name)
            if event is None:
                continue
            await self._runner.session_service.append_event(session, event)

        return _compose_prompt(prompt_body, instructions)


def _split_chat_ctx(chat_ctx: ChatContext) -> tuple[dict[str, list[str]], list[ChatMessage]]:
    instructions: dict[str, list[str]] = {"system": [], "developer": []}
    conversation: list[ChatMessage] = []

    for message in chat_ctx.messages():
        text = message.text_content
        if not text:
            continue

        if message.role in instructions:
            instructions[message.role].append(text)
        elif message.role in {"user", "assistant"}:
            conversation.append(message)

    return instructions, conversation


def _compose_prompt(prompt_body: str, instructions: dict[str, list[str]]) -> str:
    sections: list[str] = []
    if instructions["system"]:
        sections.append("System instructions:\n" + "\n\n".join(instructions["system"]))
    if instructions["developer"]:
        sections.append("Developer instructions:\n" + "\n\n".join(instructions["developer"]))

    prompt_body = prompt_body or "Continue the conversation."
    if sections:
        sections.append("User message:\n" + prompt_body)
        return "\n\n".join(sections)

    return prompt_body


def _render_transcript(messages: list[ChatMessage]) -> str:
    return "\n".join(f"{message.role}: {message.text_content}" for message in messages)


def _chat_message_to_adk_event(mods: _ADKModules, message: ChatMessage, assistant_name: str) -> Any | None:
    text = message.text_content
    if not text:
        return None

    role = "user" if message.role == "user" else "model"
    author = "user" if role == "user" else assistant_name
    return mods.Event(
        invocation_id="",
        author=author,
        content=mods.Content(role=role, parts=[_part_from_text(mods, text)]),
    )


def _part_from_text(mods: _ADKModules, text: str) -> Any:
    part_cls = mods.Part
    from_text = getattr(part_cls, "from_text", None)
    if callable(from_text):
        return from_text(text=text)
    return part_cls(text=text)


def _event_text(event: Any, *, include_thoughts: bool) -> str:
    content = getattr(event, "content", None)
    if content is None or getattr(content, "role", None) != "model":
        return ""

    parts = getattr(content, "parts", None) or []
    texts: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if not text:
            continue
        if getattr(part, "thought", False) and not include_thoughts:
            continue
        texts.append(text)

    return "".join(texts)


def _delta_text(text: str, emitted_text: str) -> tuple[str, str]:
    if text.startswith(emitted_text):
        delta = text[len(emitted_text) :]
        return delta, text

    return text, emitted_text + text
