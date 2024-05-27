from __future__ import annotations

import enum
import inspect
import typing
from typing import Any, Callable

from attrs import define, field


class UseDocMarker:
    pass


METADATA_ATTR = "__livekit_ai_metadata__"
USE_DOCSTRING = UseDocMarker()


def _set_metadata(
    f: Callable,
    name: str | None = None,
    desc: str | UseDocMarker | None = None,
    auto_retry: bool = False,
) -> None:
    if desc is None:
        desc = ""

    if isinstance(desc, UseDocMarker):
        desc = inspect.getdoc(f)
        if desc is None:
            raise ValueError(
                f"missing docstring for function {f.__name__}, "
                "use explicit description or provide docstring"
            )

    metadata = AIFncMetadata(
        name=name or f.__name__,
        desc=desc,
        auto_retry=auto_retry,
    )

    setattr(f, METADATA_ATTR, metadata)


def ai_callable(
    *,
    name: str | None = None,
    desc: str | UseDocMarker | None = None,
    auto_retry: bool = False,
) -> Callable:
    def deco(f):
        _set_metadata(f, name=name, desc=desc, auto_retry=auto_retry)
        return f

    return deco


class FunctionContext:
    def __init__(self) -> None:
        self._fncs = dict[str, AIFunction]()

        for _, member in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(member, METADATA_ATTR):
                self._register_ai_function(member)

    def ai_callable(
        self,
        *,
        name: str | None = None,
        desc: str | UseDocMarker | None = None,
        auto_retry: bool = True,
    ) -> Callable:
        def deco(f):
            _set_metadata(f, name=name, desc=desc, auto_retry=auto_retry)
            self._register_ai_function(f)

        return deco

    def _register_ai_function(self, fnc: Callable) -> None:
        if not hasattr(fnc, METADATA_ATTR):
            print(f"no metadata for {fnc}")
            return

        metadata: AIFncMetadata = getattr(fnc, METADATA_ATTR)
        if metadata.name in self._fncs:
            raise ValueError(f"duplicate AI function name: {metadata.name}")

        sig = inspect.signature(fnc)
        type_hints = typing.get_type_hints(fnc)  # Annotated[T, ...] -> T
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

        aifnc = AIFunction(metadata=metadata, fnc=fnc, args=args)
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


@define(frozen=True)
class TypeInfo:
    desc: str = ""
    choices: list[Any] = field(factory=list)


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


@define(kw_only=True, frozen=True)
class CalledFunction:
    fnc_name: str
    fnc: Callable
    args: dict


def is_type_supported(t: type) -> bool:
    if t in (str, int, float, bool):
        return True

    if issubclass(t, enum.Enum):
        return all(isinstance(e.value, str) for e in t)

    return False
