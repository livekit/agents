from typing import Literal, TypeVar, Union

_T = TypeVar("_T")


class NotGiven:
    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> str:
        return "NOT_GIVEN"


NotGivenOr = Union[_T, NotGiven]
NOT_GIVEN = NotGiven()


AgentState = Union[Literal["initializing", "listening", "thinking", "speaking"], str]
