"""Voice activity detection: the base classes a VAD plugin written in Python implements."""

from __future__ import annotations

from ._generated import public_vad as _public
from ._generated.public_vad import *  # noqa: F403

__all__: list[str] = []
__all__ += _public.__all__
