# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/llm/provider_format/host.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .chat_items import ChatItem

# Record declarations:


@dataclass
class ConvertedItem:
    """One item of a converted conversation: the item at `index`, or `rewritten` in its place."""

    index: int
    rewritten: ChatItem | None


DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE: Final = "<instructions>\n{content}\n</instructions>"
"""`convert_mid_conversation_instructions`' default wrapping; `{content}` is the only
placeholder.
"""
