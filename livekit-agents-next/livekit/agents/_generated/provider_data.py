# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/llm/provider_format/mod.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations


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


AnthropicFormatData.__annotations__ = get_annotations(AnthropicFormatData, eval_str=True)
AnthropicFormatData.__module__ = "livekit.agents.llm.utils"
BedrockFormatData.__annotations__ = get_annotations(BedrockFormatData, eval_str=True)
BedrockFormatData.__module__ = "livekit.agents.llm.utils"
GoogleFormatData.__annotations__ = get_annotations(GoogleFormatData, eval_str=True)
GoogleFormatData.__module__ = "livekit.agents.llm.utils"
MistralFormatData.__annotations__ = get_annotations(MistralFormatData, eval_str=True)
MistralFormatData.__module__ = "livekit.agents.llm.utils"
