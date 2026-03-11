"""Experimental features for the NVIDIA LiveKit plugin."""

import typing

if typing.TYPE_CHECKING:
    from . import personaplex


def __getattr__(name: str) -> typing.Any:
    if name == "personaplex":
        try:
            from . import personaplex
        except ImportError as e:
            raise ImportError(
                "The 'personaplex' module requires optional dependencies. "
                "Please install them with: pip install 'livekit-plugins-nvidia[personaplex]'"
            ) from e

        return personaplex

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["personaplex"]
