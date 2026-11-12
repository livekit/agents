"""Provider attempts for tests that only feed a request's read deadline."""

from __future__ import annotations

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, ProviderAttempts


class _NoAttempt:
    """A request whose attempt is never run: only the observation methods are exercised."""

    async def _core_attempt(self, retry: bool) -> None:
        return None

    def _core_produced(self) -> bool:
        return False

    def _core_retrying(self, error: Exception) -> None:
        pass

    def _core_failed(self, error: Exception) -> None:
        pass


def observed_attempts(
    kind: str, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
) -> ProviderAttempts:
    return ProviderAttempts(kind, conn_options, _NoAttempt())
