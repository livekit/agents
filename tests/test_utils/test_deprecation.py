from __future__ import annotations

import logging
from typing import Any

import pytest

from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils.deprecation import deprecate_params

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("deadline", "removal"),
    [
        ({}, ""),
        ({"target_version": "v2.0"}, " and will be removed in v2.0"),
        ({"target_date": "March 1, 2027"}, " and will be removed on March 1, 2027"),
        (
            {"target_version": "v2.0", "target_date": "March 1, 2027"},
            " and will be removed in v2.0 on March 1, 2027",
        ),
    ],
)
def test_deprecated_parameter_warning_includes_removal_notice(
    deadline: dict[str, Any], removal: str, caplog: pytest.LogCaptureFixture
) -> None:
    @deprecate_params({"old": "Use new instead."}, **deadline)
    def fn(old: int = 0) -> int:
        return old

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        assert fn(42) == 42

    assert caplog.messages == [f"old is deprecated{removal}. Use new instead."]


def test_omitted_parameters_do_not_warn_even_with_none_defaults(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @deprecate_params({"provider": "Omit provider."}, target_date="March 1, 2027")
    def fn(provider: str | None = None) -> str | None:
        return provider

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        assert fn() is None

    assert not caplog.messages


def test_only_given_parameters_are_grouped_by_suggestion(caplog: pytest.LogCaptureFixture) -> None:
    @deprecate_params({"first": "Use new.", "second": "Use new.", "third": "Use other."})
    def fn(
        first: NotGivenOr[int | None] = NOT_GIVEN,
        *,
        second: NotGivenOr[int] = NOT_GIVEN,
        third: NotGivenOr[int] = NOT_GIVEN,
    ) -> tuple[NotGivenOr[int | None], NotGivenOr[int], NotGivenOr[int]]:
        return first, second, third

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        assert fn(None, second=2, third=NOT_GIVEN) == (None, 2, NOT_GIVEN)

    assert caplog.messages == ["first, second are deprecated. Use new."]
