import pytest
from prompts import prompt

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("name", ["agents_sdks", "greeting", "user_away"])
def test_prompt_loads_named_prompt(name: str) -> None:
    assert prompt(name).strip()


def test_missing_prompt_raises() -> None:
    with pytest.raises(FileNotFoundError):
        prompt("no_such_prompt")
