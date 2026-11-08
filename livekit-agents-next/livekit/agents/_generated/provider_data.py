# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/llm/provider_format/mod.rs
#     rust/crates/lk-agents-abi/src/llm/provider_format/vocabulary.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations
from typing import Literal, TypeAlias

ProviderFormat: TypeAlias = Literal["openai", "openai.responses", "google", "aws", "anthropic", "mistralai"]
"""The request schema `to_provider_format` and `to_fnc_ctx` write."""


@dataclass(kw_only=True, slots=True)
class ToolSchemaRaw:
    """`info.raw_schema`, verbatim."""

    name: str
    raw_schema: object


@dataclass(kw_only=True, slots=True)
class ToolSchemaLegacy:
    """`build_legacy_openai_schema(tool, internally_tagged=True)`: for Anthropic, Bedrock, Gemini
    and the Responses API.
    """

    schema: object


@dataclass(kw_only=True, slots=True)
class ToolSchemaStrict:
    """`build_strict_openai_schema(tool)`."""

    schema: object


@dataclass(kw_only=True, slots=True)
class ToolSchemaProvider:
    """Only the Responses API accepts one."""

    schema: object


ToolSchema: TypeAlias = ToolSchemaRaw | ToolSchemaLegacy | ToolSchemaStrict | ToolSchemaProvider
"""One tool as the host hands it to `to_fnc_ctx`: the host builds the OpenAI-shaped schema, and
the reference's `strict` flag becomes the variant.
"""


@dataclass
class AnthropicFormatData:
    system_messages: list[str] | None


@dataclass
class BedrockFormatData:
    system_messages: list[str] | None


@dataclass
class GoogleFormatData:
    system_messages: list[str] | None


@dataclass
class MistralFormatData:
    instructions: str | None


TOOL_BEHAVIOR: TypeAlias = Literal["UNSPECIFIED", "BLOCKING", "NON_BLOCKING"]
"""Gemini's function `behavior`; `UNSPECIFIED` is spelled by omitting it."""


AnthropicFormatData.__annotations__ = get_annotations(AnthropicFormatData, eval_str=True)
AnthropicFormatData.__module__ = "livekit.agents.llm.utils"
BedrockFormatData.__annotations__ = get_annotations(BedrockFormatData, eval_str=True)
BedrockFormatData.__module__ = "livekit.agents.llm.utils"
GoogleFormatData.__annotations__ = get_annotations(GoogleFormatData, eval_str=True)
GoogleFormatData.__module__ = "livekit.agents.llm.utils"
MistralFormatData.__annotations__ = get_annotations(MistralFormatData, eval_str=True)
MistralFormatData.__module__ = "livekit.agents.llm.utils"
