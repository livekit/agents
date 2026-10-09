from __future__ import annotations

import json
from collections.abc import Mapping, Sequence, Set
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Protocol,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)

from .log import logger
from .utils import is_given

if TYPE_CHECKING:
    from .stt import STT
    from .tts import TTS
    from .vad import VAD


_T = TypeVar("_T")
Sensitive = Annotated[_T, "sensitive"]
"""Marks an option for exclusion from session reports."""


def reportable_option_names(*option_types: type) -> frozenset[str]:
    """Return declared fields, excluding any marked sensitive in the supplied types."""
    declared: set[str] = set()
    sensitive: set[str] = set()
    for option_type in option_types:
        for name, annotation in get_type_hints(option_type, include_extras=True).items():
            declared.add(name)
            if get_origin(annotation) is Annotated and "sensitive" in get_args(annotation)[1:]:
                sensitive.add(name)
    return frozenset(declared - sensitive)


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

    Return the options worth reporting, keyed by name; values can be primitives, mappings
    or sequences of them. Leave secrets and endpoints out: the report is uploaded. Objects
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
    return _describe_option_object(value)


def component_metadata(component: STT | TTS | VAD) -> dict[str, str]:
    try:
        return {"model": component.model, "provider": component.provider}
    except Exception:
        logger.debug("component metadata failed on %s", type(component).__name__, exc_info=True)
        return {}


def snapshot_component(component: STT | TTS | VAD) -> dict[str, Any]:
    """Snapshot one component without letting its description failure affect siblings."""
    cls = type(component)
    options: dict[str, Any] = component_metadata(component)
    try:
        options.update(
            _serialize_option_value(
                {
                    key: value
                    for key, value in component.describe_options().items()
                    if value is not None and is_given(value)
                }
            )
        )
    except Exception:
        logger.debug("describe_options() failed on %s", cls.__name__, exc_info=True)
    return {**options, "type": f"{cls.__module__}.{cls.__name__}"}
