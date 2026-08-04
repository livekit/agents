"""Test that APIConnectionError names the underlying failure when stringified,
which is the only detail available to callers that serialize the error."""

from __future__ import annotations

import pytest

from livekit.agents import APIConnectionError, APITimeoutError

pytestmark = pytest.mark.unit


def _wrap(inner: BaseException, err: APIConnectionError | None = None) -> APIConnectionError:
    """`raise ... from inner` sets __cause__ after construction; assign it the same way."""
    err = err or APIConnectionError()
    err.__cause__ = inner
    return err


def test_str_without_cause_is_the_message() -> None:
    assert str(APIConnectionError()) == "Connection error."
    assert str(APIConnectionError("failed to connect")) == "failed to connect"


def test_str_names_the_cause() -> None:
    err = _wrap(ValueError("payload is not completed"))
    assert str(err) == "Connection error. (caused by ValueError: payload is not completed)"


def test_str_reports_the_root_of_the_chain() -> None:
    """Providers wrap transport failures behind a placeholder message of their own."""
    root = OSError("peer closed connection")
    middle = RuntimeError("Connection error.")
    middle.__cause__ = root

    assert "OSError: peer closed connection" in str(_wrap(middle))


def test_str_survives_a_cause_whose_str_raises() -> None:
    class Hostile(Exception):
        def __str__(self) -> str:
            raise RuntimeError("nope")

    assert str(_wrap(Hostile())) == "Connection error. (caused by Hostile)"


def test_str_terminates_on_a_cyclic_chain() -> None:
    a, b = ValueError("a"), ValueError("b")
    a.__cause__, b.__cause__ = b, a

    assert "ValueError" in str(_wrap(a))


def test_str_terminates_when_the_error_causes_itself() -> None:
    err = APIConnectionError("failed to connect")
    err.__cause__ = err

    assert str(err) == "failed to connect"


def test_str_terminates_on_a_cycle_between_two_connection_errors() -> None:
    outer, inner = APIConnectionError("outer"), APIConnectionError("inner")
    outer.__cause__, inner.__cause__ = inner, outer

    assert str(outer) == "outer (caused by APIConnectionError: inner)"


def test_str_stops_after_ten_distinct_causes() -> None:
    """The 10th is named on its own; nothing deeper is walked or nested into the message."""
    root: BaseException = OSError("peer closed connection")
    for i in range(20):
        wrapper = APIConnectionError(f"wrapped {i}")
        wrapper.__cause__ = root
        root = wrapper

    assert str(_wrap(root)) == "Connection error. (caused by APIConnectionError: wrapped 10)"


def test_timeout_subclass_keeps_its_own_message() -> None:
    err = _wrap(TimeoutError(), APITimeoutError())
    assert str(err).startswith("Request timed out. (caused by TimeoutError")


def test_raise_from_populates_the_message() -> None:
    """The real statement used at every wrap site."""
    with pytest.raises(APIConnectionError) as exc_info:
        try:
            raise OSError("connection reset by peer")
        except OSError as e:
            raise APIConnectionError(retryable=False) from e

    assert str(exc_info.value) == "Connection error. (caused by OSError: connection reset by peer)"
