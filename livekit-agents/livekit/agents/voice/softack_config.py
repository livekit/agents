"""Configuration module for soft-ack detection.

This module manages soft-ack settings that can be configured via environment variables
and provides utilities for soft-ack validation.
"""

import os
import string
from typing import Set
from pathlib import Path

# Load .env file to ensure environment variables are available
try:
    from dotenv import load_dotenv
    
    env_path = None
    current_dir = Path.cwd()
    
    # Search strategy:
    # 1. Current working directory
    # 2. Parent directories up to 5 levels
    # 3. Known locations: examples/voice_agents/, examples/
    
    search_locations = [current_dir] + list(current_dir.parents[:10])
    
    for search_dir in search_locations:
        # Direct .env in this directory
        candidate = search_dir / '.env'
        if candidate.exists():
            env_path = candidate
            break
        
        # Check examples/voice_agents/.env
        candidate = search_dir / 'examples' / 'voice_agents' / '.env'
        if candidate.exists():
            env_path = candidate
            break
        
        # Check examples/.env.example as fallback
        candidate = search_dir / 'examples' / '.env.example'
        if candidate.exists():
            env_path = candidate
            break
    
    if env_path:
        load_dotenv(env_path)
    else:
        # No .env file found, try default load_dotenv() behavior
        load_dotenv()
        
except ImportError:
    # dotenv not installed, skip loading (env vars might be set via other means)
    pass


def _load_soft_acks_from_env() -> Set[str]:
    """Load soft-ack set from environment variable.
    
    Expected format in .env:
    LIVEKIT_SOFT_ACKS="okay,yeah,uh-huh,ok,hmm,right"
    
    Returns:
        Set of lowercase soft-ack words. Defaults to standard set if env var not set.
    """
    import logging #c
    logger = logging.getLogger("livekit.agents.voice.softacks") #c
    
    env_value = os.getenv("LIVEKIT_SOFT_ACKS", "").strip()
    
    logger.info(f"[SOFTACK_CONFIG] LIVEKIT_SOFT_ACKS env var: '{env_value}'") #c
    
    if env_value:
        # Parse comma-separated values and normalize
        soft_acks = {item.strip().lower() for item in env_value.split(",") if item.strip()}
        if soft_acks:  # Only use if non-empty
            logger.info(f"[SOFTACK_CONFIG] Loaded custom soft-acks from env: {soft_acks}")#c
            return soft_acks
    
    # Default soft-ack set
    default_set = {"okay", "yeah", "uhhuh", "ok", "hmm", "right"} #c
    logger.info(f"[SOFTACK_CONFIG] Using default soft-acks: {default_set}") #c
    return default_set #c


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
