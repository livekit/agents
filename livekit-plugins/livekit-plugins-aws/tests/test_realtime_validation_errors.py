from types import SimpleNamespace

from livekit.plugins.aws.experimental.realtime.realtime_model import (
    _is_recoverable_validation_error,
)


def test_system_instability_validation_error_is_recoverable() -> None:
    exc = SimpleNamespace(message="System instability detected. Please retry your request.")

    assert _is_recoverable_validation_error(exc) is True


def test_unrecognized_validation_error_is_not_recoverable() -> None:
    exc = SimpleNamespace(message="The provided request is invalid.")

    assert _is_recoverable_validation_error(exc) is False
