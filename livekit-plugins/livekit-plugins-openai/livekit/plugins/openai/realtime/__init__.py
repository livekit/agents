import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import gpt_live_types as gpt_live_types
    from .gpt_live_model import (
        GPTLiveDelegation,
        GPTLiveModel,
        GPTLiveVoices,
        GPTVoiceSession,
        ResponsesDelegationOptions,
    )
    from .realtime_model import RealtimeModel, RealtimeSession


_EXPORTS = {
    "gpt_live_types": ("gpt_live_types", None),
    "RealtimeModel": ("realtime_model", "RealtimeModel"),
    "RealtimeSession": ("realtime_model", "RealtimeSession"),
    **{
        name: ("gpt_live_model", name)
        for name in (
            "GPTLiveModel",
            "GPTVoiceSession",
            "GPTLiveDelegation",
            "GPTLiveVoices",
            "ResponsesDelegationOptions",
        )
    },
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = target
    loaded = importlib.import_module(f".{module}", __name__)
    value = getattr(loaded, attribute) if attribute else loaded
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "RealtimeSession",
    "RealtimeModel",
    "GPTLiveModel",
    "GPTVoiceSession",
    "GPTLiveDelegation",
    "GPTLiveVoices",
    "ResponsesDelegationOptions",
]

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
