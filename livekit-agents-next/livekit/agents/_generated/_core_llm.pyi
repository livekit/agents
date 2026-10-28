# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` star-imports it.

from __future__ import annotations

from typing import Any

from livekit.agents.llm import LLM
from livekit.agents.types import APIConnectOptions

from .._core import ChatItem

class InferenceLlm:
    """The gateway LLM behind `inference.LLM`. Its options are shared by every view
    (`identified`); each view's requests carry its own job's identification.
    """
    def __init__(
        self,
        model: str,
        *,
        provider: str | None = None,
        base_url: str,
        api_key: str,
        api_secret: str,
        inference_class: str | None = None,
        extra_kwargs: dict[str, object] | None = None,
    ) -> None:
        """A client of `model` on the gateway at `base_url`. Starts nothing."""
    def identified(
        self,
        headers: list[tuple[str, str]],
        reporter: LLM[Any] | None = None,
    ) -> InferenceLlm:
        """This client as a session of the job `headers` (`get_inference_headers()`) identifies
        holds it: the same options, its requests carrying that job. `reporter` is the host
        `llm.LLM` whose `error` events report the attempts this view retries.
        """
    @property
    def model(self) -> str: ...
    @property
    def extra_kwargs(self) -> dict[str, object]:
        """The persistent request options, a copy."""
    def update_options(
        self,
        *,
        model: str | None = None,
        extra_kwargs: dict[str, object] | None = None,
    ) -> None:
        """Read by the next `chat()`. `extra_kwargs` replaces the persistent options."""
    def chat(
        self,
        items: list[ChatItem],
        *,
        tools: list[dict[str, object]],
        conn_options: APIConnectOptions,
        parallel_tool_calls: bool | None = None,
        tool_choice: str | dict[str, object] | None = None,
        response_format: object | None = None,
        extra_kwargs: dict[str, object] | None = None,
        headers: list[tuple[str, str]] | None = None,
    ) -> InferenceLlmChunks:
        """Opens a completion of `items`; no I/O until the first pull. `headers` identify the
        caller's job, read in its context.
        """

class InferenceLlmChunks:
    async def recv(self) -> dict[str, object] | None: ...
    async def aclose(self) -> None: ...
