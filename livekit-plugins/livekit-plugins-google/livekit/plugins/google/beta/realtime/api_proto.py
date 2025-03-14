from __future__ import annotations

from typing import Literal, Sequence, Union

from google.genai import types

from ..._utils import _build_gemini_ctx, _build_tools

LiveAPIModels = Literal["gemini-2.0-flash-exp"]

Voice = Literal["Puck", "Charon", "Kore", "Fenrir", "Aoede"]

__all__ = ["_build_tools", "ClientEvents", "_build_gemini_ctx"]

ClientEvents = Union[
    types.ContentListUnion,
    types.ContentListUnionDict,
    types.LiveClientContentOrDict,
    types.LiveClientRealtimeInput,
    types.LiveClientRealtimeInputOrDict,
    types.LiveClientToolResponseOrDict,
    types.FunctionResponseOrDict,
    Sequence[types.FunctionResponseOrDict],
]
