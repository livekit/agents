from __future__ import annotations

from typing import Dict

from . import __version__

GOTRUE_URL = "http://localhost:9999"
DEFAULT_HEADERS: Dict[str, str] = {
    "X-Client-Info": f"gotrue-py/{__version__}",
}
EXPIRY_MARGIN = 10  # seconds
MAX_RETRIES = 10
RETRY_INTERVAL = 2  # deciseconds
STORAGE_KEY = "supabase.auth.token"
