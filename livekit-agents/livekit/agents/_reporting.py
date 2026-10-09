from __future__ import annotations

import json
from collections.abc import Collection, Iterator, Mapping, Sequence, Set
from dataclasses import fields, is_dataclass
from enum import Enum
from types import UnionType
from typing import (
    Annotated,
    Any,
    Protocol,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)

from pydantic import BaseModel
from typing_extensions import NotRequired, Required, is_typeddict

from .log import logger
from .utils import is_given

_T = TypeVar("_T")
Sensitive = Annotated[_T, "sensitive"]
"""Marks an option for exclusion from session reports."""


def report_options(
    config: Any, *option_types: type, exclude: Collection[str] | Mapping[str, Any] = ()
) -> dict[str, Any]:
    """Read declared config fields recursively, omitting Sensitive, None and NOT_GIVEN values.

    Dataclasses and Pydantic models carry their schema. Dictionaries require TypedDict
    schemas, supplied explicitly or through a parent field's annotation. Undeclared fields
    are omitted. Exclusions can be field names or a mapping of nested exclusions.
    """
    if config is None or not is_given(config):
        return {}
    if is_dataclass(config) and not isinstance(config, type):
        option_types = (type(config),)
        values = {field.name: getattr(config, field.name) for field in fields(config)}
    elif isinstance(config, BaseModel):
        option_types = (type(config),)
        values = {name: getattr(config, name) for name in type(config).model_fields}
    else:
        values = config
    annotations: dict[str, list[Any]] = {}
    for option_type in option_types:
        hints = (
            {name: field.rebuild_annotation() for name, field in option_type.model_fields.items()}
            if issubclass(option_type, BaseModel)
            else get_type_hints(option_type, include_extras=True)
        )
        for name, annotation in hints.items():
            annotations.setdefault(name, []).extend(_unwrap_option_types(annotation))

    exclusions = exclude if isinstance(exclude, Mapping) else dict.fromkeys(exclude, True)
    result: dict[str, Any] = {}
    for key, value in values.items():
        if key not in annotations or value is None or not is_given(value):
            continue
        if any(
            get_origin(annotation) is Annotated and "sensitive" in get_args(annotation)[1:]
            for annotation in annotations[key]
        ):
            continue
        nested_exclude = exclusions.get(key, ())
        if nested_exclude is True:
            continue
        result[key] = _report_option_value(value, annotations[key], exclude=nested_exclude or ())
    return result


def _unwrap_option_types(annotation: Any) -> Iterator[Any]:
    yield annotation
    origin = get_origin(annotation)
    if origin in (Annotated, Required, NotRequired):
        yield from _unwrap_option_types(get_args(annotation)[0])
    elif origin in (Union, UnionType):
        for argument in get_args(annotation):
            yield from _unwrap_option_types(argument)


def _report_option_value(
    value: Any, annotations: list[Any], *, exclude: Collection[str] | Mapping[str, Any]
) -> Any:
    if (is_dataclass(value) and not isinstance(value, type)) or isinstance(value, BaseModel):
        return report_options(value, exclude=exclude)
    if isinstance(value, Mapping):
        schemas = [annotation for annotation in annotations if is_typeddict(annotation)]
        return report_options(value, *schemas, exclude=exclude)
    if isinstance(value, (Sequence, Set)) and not isinstance(value, (str, bytes)):
        item_annotations = [
            item_type
            for annotation in annotations
            if get_origin(annotation) in (list, tuple, set, frozenset, Sequence, Set)
            for argument in get_args(annotation)
            for item_type in _unwrap_option_types(argument)
        ]
        items = sorted(value, key=str) if isinstance(value, Set) else value
        return [
            _report_option_value(item, item_annotations, exclude=exclude)
            for item in items
            if item is not None and is_given(item)
        ]
    return value


_SESSION_OPTION_KEY_ALIASES = {
    "keyterms": "lk.pii.keyterms",
}

# Option keys never written to the report: prompt text authored by the customer
# (``stt_context_options.keyterm_detection.instructions``) can embed anything about their
# business or users, and the report has no use for it.
_SESSION_OPTION_OMITTED_KEYS = frozenset({"instructions"})


_OPTION_PRIMITIVES = (str, bool, int, float)


@runtime_checkable
class DescribesOptions(Protocol):
    """An object that can appear in ``AgentSession`` options (a turn detector, a model) and
    wants the session report to show its configuration.

    Return the options worth reporting, keyed by name; values can be primitives, models,
    mappings or sequences of them. Leave secrets and endpoints out: the report is uploaded. Objects
    without this method are reported by class name alone."""

    def describe_options(self) -> Mapping[str, Any]: ...


def _describe_option_object(obj: object) -> str:
    """Render an object from the session options as ``module.Class`` or, when it implements
    :class:`DescribesOptions`, ``module.Class(k=v, ...)``.

    The OTel log exporter stringifies anything that is not a primitive, which for these
    objects yields the default ``<... object at 0x...>`` repr. The class alone is stable and
    safe; the object itself decides what else is worth showing."""
    cls = type(obj)
    name = f"{cls.__module__}.{cls.__name__}"
    describe = getattr(obj, "describe_options", None)
    if not callable(describe):
        return name
    try:
        options = describe()
    except Exception:
        logger.debug("describe_options() failed on %s", name, exc_info=True)
        return name
    parts: list[str] = []
    for key, value in options.items():
        if value is None or not is_given(value):
            continue
        rendered = (
            str(value)
            if isinstance(value, _OPTION_PRIMITIVES)
            else json.dumps(_serialize_option_value(value), sort_keys=True, default=str)
        )
        parts.append(f"{key}={rendered}")
    return f"{name}({', '.join(parts)})"


def _serialize_option_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return _serialize_option_value(value.value)
    if value is None or isinstance(value, _OPTION_PRIMITIVES):
        return value
    if isinstance(value, Mapping):
        return {
            _SESSION_OPTION_KEY_ALIASES.get(k, k): _serialize_option_value(v)
            for k, v in value.items()
            if k not in _SESSION_OPTION_OMITTED_KEYS and is_given(v)
        }
    if isinstance(value, (Sequence, Set)) and not isinstance(value, (str, bytes)):
        # any Sequence is a valid option value (tts_text_transforms accepts one), so
        # serialize the elements rather than collapsing the container to its class name
        items = sorted(value, key=str) if isinstance(value, Set) else value
        return [_serialize_option_value(v) for v in items]

    from .llm import LLM, DuplexModel, RealtimeModel
    from .stt import STT
    from .tts import TTS
    from .vad import VAD

    if isinstance(value, (LLM, RealtimeModel, DuplexModel, STT, TTS, VAD)):
        cls = type(value)
        options: dict[str, Any] = {}
        try:
            options.update(
                _serialize_option_value(
                    {
                        key: option
                        for key, option in value.describe_options().items()
                        if option is not None and is_given(option)
                    }
                )
            )
        except Exception:
            logger.debug("describe_options() failed on %s", cls.__name__, exc_info=True)
        try:
            options.update(model=value.model, provider=value.provider)
            if isinstance(value, TTS):
                options.update(TTS.describe_options(value))
        except Exception:
            logger.debug("model metadata failed on %s", cls.__name__, exc_info=True)
        return {**options, "type": f"{cls.__module__}.{cls.__name__}"}
    return _describe_option_object(value)
