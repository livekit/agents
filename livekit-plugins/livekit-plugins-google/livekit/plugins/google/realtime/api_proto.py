from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Union

from google.genai import types

# Gemini API deprecations: https://ai.google.dev/gemini-api/docs/deprecations
# Gemini API release notes with preview deprecations: https://ai.google.dev/gemini-api/docs/changelog
# live models: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api
# VertexAI retirement: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions#retired-models
# Additional references:
# 1. https://github.com/kazunori279/adk-streaming-test/blob/main/test_report.md
LiveAPIModels = Literal[
    # VertexAI models
    "gemini-live-2.5-flash-native-audio",  # GA https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash-live-api#live-2.5-flash
    "gemini-live-2.5-flash-preview-native-audio-09-2025",  # Public preview https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash-live-api#live-2.5-flash-preview
    "gemini-live-2.5-flash-preview-native-audio",  # still works, possibly an alias, but not mentioned in any docs or changelog
    # Gemini API models
    "gemini-2.5-flash-native-audio-preview-12-2025",  # https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash-live
    "gemini-2.5-flash-native-audio-preview-09-2025",  # https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash-live
    "gemini-2.0-flash-exp",  # still works in Gemini API but not VertexAI
]

Voice = Literal[
    "Achernar",
    "Achird",
    "Algenib",
    "Algieba",
    "Alnilam",
    "Aoede",
    "Autonoe",
    "Callirrhoe",
    "Charon",
    "Despina",
    "Enceladus",
    "Erinome",
    "Fenrir",
    "Gacrux",
    "Iapetus",
    "Kore",
    "Laomedeia",
    "Leda",
    "Orus",
    "Pulcherrima",
    "Puck",
    "Rasalgethi",
    "Sadachbia",
    "Sadaltager",
    "Schedar",
    "Sulafat",
    "Umbriel",
    "Vindemiatrix",
    "Zephyr",
    "Zubenelgenubi",
]


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
