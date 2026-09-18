"""Handing reasoning and tool use to an expert.

The conversation model keeps the user talking and gains one tool, ``lk_agents_delegate``.
Everything needing reasoning, lookups or actions goes to the expert, which returns facts the
conversation phrases. The expert is reached over A2A, and the endpoint need not be ours.

Needs the ``a2a`` extra: ``pip install 'livekit-agents[a2a]'``.
"""

from typing import TYPE_CHECKING, Any

from .delegate import (
    DELEGATE_TOOL_NAME,
    Delegate,
    DelegateStream,
    DelegationOptions,
    resolve_delegation_options,
)

if TYPE_CHECKING:
    from .a2a import A2ADelegate
    from .tool import build_delegate_tool

# core reads the options from this package, so nothing that needs the a2a extra may load
# with it; what does is resolved on first use instead
_LAZY = {"A2ADelegate": ".a2a", "build_delegate_tool": ".tool"}


def __getattr__(name: str) -> Any:
    if (module := _LAZY.get(name)) is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


__all__ = [
    "DELEGATE_TOOL_NAME",
    "A2ADelegate",
    "Delegate",
    "DelegateStream",
    "DelegationOptions",
    "build_delegate_tool",
    "resolve_delegation_options",
]
