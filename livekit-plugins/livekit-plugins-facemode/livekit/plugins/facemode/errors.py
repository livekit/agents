class FaceModeError(RuntimeError):
    """Base error for FaceMode integration failures."""


class FaceModeAPIError(FaceModeError):
    """Raised when the FaceMode REST API rejects a request."""


class FaceModeProtocolError(FaceModeError):
    """Raised when the FaceMode WebSocket reports a protocol error."""


class FaceModeNotReadyError(FaceModeError):
    """Raised when the avatar does not become ready in time."""
