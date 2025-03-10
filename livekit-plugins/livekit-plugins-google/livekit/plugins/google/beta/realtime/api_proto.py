from __future__ import annotations

from typing import Literal, Sequence, Union

from google.genai import types

LiveAPIModels = Literal["gemini-2.0-flash-001"]

Voice = Literal["Puck", "Charon", "Kore", "Fenrir", "Aoede"]


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
