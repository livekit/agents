from __future__ import annotations

_default_client: _60dbClient | None = None


class _60dbClient:
    """Client for authenticating with 60db.ai services.

    Instantiating this sets the global default client, so that STT/TTS/LLM
    constructors can pick up the API key without explicit arguments or env vars.
    """

    def __init__(self, api_key: str) -> None:
        global _default_client
        self._api_key = api_key
        _default_client = self

    @property
    def api_key(self) -> str:
        return self._api_key


def _get_default_api_key() -> str | None:
    """Return the API key from the global client, if set."""
    if _default_client is not None:
        return _default_client.api_key
    return None
