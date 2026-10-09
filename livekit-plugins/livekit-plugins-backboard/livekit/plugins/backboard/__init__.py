"""
Backboard.io plugin for LiveKit Agents.

Provides an LLM integration with Backboard's persistent memory,
RAG document retrieval, and 1,800+ model backend.

Usage::

    from livekit.plugins import backboard

    session = AgentSession(
        llm=backboard.LLM(
            assistant_id="your-assistant-id",
            llm_provider="openai",
            model_name="gpt-4o",
        ),
        stt=...,
        tts=...,
    )

Environment Variables:
    BACKBOARD_API_KEY: Your Backboard.io API key.
"""

from .llm import LLM, BackboardLLM, BackboardLLMStream
from .session import SessionStore
from .version import __version__

__all__ = [
    "LLM",
    "BackboardLLM",
    "BackboardLLMStream",
    "SessionStore",
    "__version__",
]
