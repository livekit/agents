from types import SimpleNamespace

from livekit.plugins.aws.experimental.realtime.realtime_model import (
    _is_recoverable_validation_error,
)


def test_realtime_package_imports_against_current_bedrock_sdk() -> None:
    """The realtime extra must import on aws-sdk-bedrock-runtime 0.10+ / 0.11.

    0.10 dropped ``Config`` and 0.11 dropped ``BedrockRuntimeClient``, so the old
    top-level imports failed before a session was created. Regression for
    https://github.com/livekit/agents/issues/6994.
    """
    from livekit.plugins.aws.experimental.realtime import RealtimeModel

    assert RealtimeModel.__name__ == "RealtimeModel"


def test_system_instability_validation_error_is_recoverable() -> None:
    exc = SimpleNamespace(message="System instability detected. Please retry your request.")

    assert _is_recoverable_validation_error(exc) is True


def test_unrecognized_validation_error_is_not_recoverable() -> None:
    exc = SimpleNamespace(message="The provided request is invalid.")

    assert _is_recoverable_validation_error(exc) is False
