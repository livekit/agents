from google.genai.types import (
    GoogleMaps,
    GoogleSearch,
    GoogleSearchRetrieval,
    ToolCodeExecution,
    UrlContext,
)

LLMTool = GoogleSearchRetrieval | ToolCodeExecution | GoogleSearch | UrlContext | GoogleMaps
