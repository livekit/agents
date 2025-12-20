"""Configuration module for soft-ack detection.

This module manages soft-ack settings that can be configured via environment variables
and provides utilities for soft-ack validation.
"""

import os
import string
from typing import Set


def _load_soft_acks_from_env() -> Set[str]:
    """Load soft-ack set from environment variable.
    
    Expected format in .env:
    LIVEKIT_SOFT_ACKS="okay,yeah,uh-huh,ok,hmm,right"
    
    Returns:
        Set of lowercase soft-ack words. Defaults to standard set if env var not set.
    """
    env_value = os.getenv("LIVEKIT_SOFT_ACKS", "").strip()
    
    if env_value:
        # Parse comma-separated values and normalize
        soft_acks = {item.strip().lower() for item in env_value.split(",") if item.strip()}
        if soft_acks:  # Only use if non-empty
            return soft_acks
    
    # Default soft-ack set
    return {"okay", "yeah", "uhhuh", "ok", "hmm", "right"}


# Global soft-ack set loaded at module import
SOFT_ACK_SET: Set[str] = _load_soft_acks_from_env()


def is_soft_ack(text: str) -> bool:
    """Check if text is a recognized soft-ack.
    
    Args:
        text: The text to check (will be lowercased and normalized).
        
    Returns:
        True if text is in the soft-ack set, False otherwise.
    """
    # Normalize: lowercase, strip whitespace, and remove punctuation
    normalized = text.lower().strip()
    normalized = normalized.translate(str.maketrans('', '', string.punctuation))
    return normalized in SOFT_ACK_SET


def reload_soft_acks() -> None:
    """Reload soft-ack configuration from environment.
    
    Call this if environment variables have been changed at runtime.
    This is primarily useful for testing.
    """
    global SOFT_ACK_SET
    SOFT_ACK_SET = _load_soft_acks_from_env()
