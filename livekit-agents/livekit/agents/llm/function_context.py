from __future__ import annotations

import abc
import enum
import inspect
import typing
from typing import Any, Callable

from attrs import define

METADATA_ATTR = "__livekit_ai_metadata__"


def ai_callable(
    *,
    name: str | None = None,
    desc: str | None = None,
    auto_retry: bool = True,
) -> Callable:
    def deco(f):
        metadata = AIFncMetadata(
            name=name or f.__name__,
            desc=desc or "",
            auto_retry=auto_retry,
        )

        setattr(f, METADATA_ATTR, metadata)
        return f

    return deco


@define(frozen=True)
class TypeInfo:
    desc: str = ""


class FunctionContext(abc.ABC):
    def __init__(self) -> None:
        self._fncs = dict[str, AIFunction]()

        # retrieve ai functions & metadata
        for _, member in inspect.getmembers(self, predicate=inspect.ismethod):
            if not hasattr(member, METADATA_ATTR):
                continue

            metadata: AIFncMetadata = getattr(member, METADATA_ATTR)
            if metadata.name in self._fncs:
                raise ValueError(f"Duplicate function name: {metadata.name}")

            sig = inspect.signature(member)
            type_hints = typing.get_type_hints(member)  # Annotated[T, ...] -> T
            args = dict()

            for name, param in sig.parameters.items():
                if param.kind not in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ):
                    raise ValueError(
                        f"unsupported parameter kind inside ai_callable: {param.kind}"
                    )

                if param.annotation is inspect.Parameter.empty:
                    raise ValueError(f"missing type annotation for parameter {name}")

                th = type_hints[name]

                if not is_type_supported(th):
                    raise ValueError(f"unsupported type {th} for parameter {name}")

                default = param.default

                type_info = _find_param_type_info(param.annotation)
                desc = type_info.desc if type_info else ""

                args[name] = AIFncArg(
                    name=name,
                    desc=desc,
                    type=th,
                    default=default,
                )

            aifnc = AIFunction(metadata=metadata, fnc=member, args=args)
            self._fncs[metadata.name] = aifnc

    @property
    def ai_functions(self) -> dict[str, AIFunction]:
        return self._fncs


def _find_param_type_info(annotation: type) -> TypeInfo | None:
    if typing.get_origin(annotation) is not typing.Annotated:
        return None

    for a in typing.get_args(annotation):
        if isinstance(a, TypeInfo):
            return a

    return None


# internal metadata
@define(kw_only=True, frozen=True)
class AIFncMetadata:
    name: str = ""
    desc: str = ""
    auto_retry: bool = True


@define(kw_only=True, frozen=True)
class AIFncArg:
    name: str
    desc: str
    type: type
    default: Any


@define(kw_only=True, frozen=True)
class AIFunction:
    metadata: AIFncMetadata
    fnc: Callable
    args: dict[str, AIFncArg]


def is_type_supported(t: type) -> bool:
    if t in (str, int, float, bool):
        return True

    if issubclass(t, enum.Enum):
        return all(isinstance(e.value, str) for e in t)

    return False
