from typing import Union

from google.genai.types import (
    GoogleMaps,
    GoogleSearch,
    GoogleSearchRetrieval,
    ToolCodeExecution,
    UrlContext,
)

_LLMTool = Union[GoogleSearchRetrieval, ToolCodeExecution, GoogleSearch, UrlContext, GoogleMaps]
