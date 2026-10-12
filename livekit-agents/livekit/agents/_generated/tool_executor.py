# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/llm/tool_executor.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from ..voice.tool_executor import DuplicatePromptArgs, ReplyPromptArgs, UpdatePromptArgs


class AsyncToolOptions(TypedDict, total=False):
    """System-message templates injected around async tool dispatch.

    Each field is either a ``str.format()`` template or a callable returning a string,
    with the args typed as ``UpdatePromptArgs`` / ``DuplicatePromptArgs`` /
    ``ReplyPromptArgs``. Unmentioned keys keep their defaults.
    """

    update_template: str | Callable[[UpdatePromptArgs], str]
    """Wraps a user-provided ``ctx.update(message)`` string before it lands in chat_ctx."""
    duplicate_reject_template: str | Callable[[DuplicatePromptArgs], str]
    """Sent to the LLM when ``on_duplicate='reject'`` blocks a duplicate call."""
    duplicate_confirm_template: str | Callable[[DuplicatePromptArgs], str]
    """Sent to the LLM when ``on_duplicate='confirm'`` requires re-call with confirmation."""
    reply_at_tail_template: str | Callable[[ReplyPromptArgs], str]
    """Instruction for the deferred reply when the pending update is still the tail of chat_ctx."""
    reply_maybe_covered_template: str | Callable[[ReplyPromptArgs], str]
    """Instruction for the deferred reply when newer items came after the pending update."""


class ToolHandlingOptions(TypedDict, total=False):
    """Configuration for the tool handling system.

    Can be passed as a plain dict::

        AgentSession(
            tool_handling={
                "async_options": {"update_template": "..."},
            },
        )

    Set on ``AgentSession``, ``Agent``, or ``AsyncToolset`` (most specific wins).
    """

    async_options: AsyncToolOptions
    """Templates injected around async tool dispatch (``ctx.update()``, duplicate
    handling, coalesced replies). Unmentioned keys keep their defaults.
    """


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
AsyncToolOptions.__module__ = "livekit.agents.voice.tool_executor"
ToolHandlingOptions.__module__ = "livekit.agents.voice.tool_executor"
