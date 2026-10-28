from datetime import date

import pytest

from livekit.agents.beta.workflows import normalize_date_of_birth

pytestmark = pytest.mark.unit


def test_two_digit_birth_years_resolve_to_the_past() -> None:
    assert normalize_date_of_birth(90, 5, 15) == date(1990, 5, 15)
    assert normalize_date_of_birth(5, 3, 1) == date(2005, 3, 1)
