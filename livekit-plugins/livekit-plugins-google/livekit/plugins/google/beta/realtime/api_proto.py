from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Union

from google.genai import types

LiveAPIModels = Literal["gemini-2.0-flash-exp"]

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
