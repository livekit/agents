# Copyright 2024-2026, Daily
#
# SPDX-License-Identifier: Apache-2.0

"""LiteLLM plugin for LiveKit Agents.

Provides access to 100+ LLM providers through the LiteLLM SDK.
"""

from .llm import LLM
from .version import __version__

__all__ = ["LLM", "__version__"]
