class FaceMarketError(Exception):
    """Base exception for the FaceMarket avatar plugin."""


class FaceMarketSessionError(FaceMarketError):
    """Raised when session lifecycle usage is invalid."""


class FaceMarketPlatformError(FaceMarketError):
    """Raised when FaceMarket platform APIs return an error."""


class SessionReadyTimeoutError(FaceMarketSessionError):
    """Raised when the coordinator never confirms session readiness."""
