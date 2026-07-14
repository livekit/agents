"""The agent's authored language, loaded like application templates."""

from functools import cache
from importlib.resources import files


@cache
def prompt(name: str) -> str:
    resource = files(__package__).joinpath(f"{name}.md")
    if not resource.is_file():
        raise FileNotFoundError(f"no prompt named {name!r}")
    return resource.read_text(encoding="utf-8")
