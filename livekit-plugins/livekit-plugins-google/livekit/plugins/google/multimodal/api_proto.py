from __future__ import annotations

from typing import Literal, Sequence, Union

from google.genai import types

MultimodalModels = Literal["gemini-2.0-flash-exp"]

Voice = Literal["Puck", "Charon", "Kore", "Fenrir", "Aoede"]
ResponseModality = Literal["AUDIO", "TEXT"]


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
