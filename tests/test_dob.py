from datetime import date

import pytest

from livekit.agents.beta.workflows.dob import normalize_date_of_birth

pytestmark = pytest.mark.unit


def test_dob_two_digit_year_normalized() -> None:
    assert normalize_date_of_birth(90, 5, 15) == date(1990, 5, 15)
    assert normalize_date_of_birth(5, 3, 1) == date(2005, 3, 1)


def test_dob_four_digit_year_unchanged() -> None:
    assert normalize_date_of_birth(1962, 7, 4) == date(1962, 7, 4)
    assert normalize_date_of_birth(2001, 12, 31) == date(2001, 12, 31)
