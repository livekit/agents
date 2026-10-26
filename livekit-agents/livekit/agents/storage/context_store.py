"""Explicit persistence of marked application-context fields in a managed turn."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import MISSING, field, fields, is_dataclass
from typing import Any, TypeVar, get_type_hints, overload

from pydantic import JsonValue, TypeAdapter

from .protocol import NotSerializable, VersionConflict, dump_state

T = TypeVar("T")
_FIELD_MARKER = "livekit.persisted"
_STORAGE_KEY = "context_fields"


@overload
def persisted(*, default_factory: Callable[[], T]) -> T: ...


@overload
def persisted(*, default: T) -> T: ...


def persisted(*, default: Any = MISSING, default_factory: Any = MISSING) -> Any:
    """Mark a dataclass field for explicit context loading and saving."""
    return field(default=default, default_factory=default_factory, metadata={_FIELD_MARKER: True})


class ContextStore:
    """A turn-scoped view of persisted context fields.

    Saves stage a validated snapshot for the host's fenced transaction. Loading
    and saving are explicit; the host owns the connection and commit lifecycle.
    """

    def __init__(self, state: JsonValue = None, *, read_only: bool = False) -> None:
        self._state = copy.deepcopy(state)
        self._committed_state = copy.deepcopy(state)
        self._read_only = read_only
        self._closed = False
        self._pending_state: JsonValue = None
        self._revisions: dict[str, int] = {}
        self._loaded: dict[int, tuple[object, dict[str, int]]] = {}

    async def load_context(self, context: T) -> T:
        """Restore marked fields, preserving runtime-only objects and declared defaults."""
        self._ensure_open()
        declared = _declared_fields(context)
        stored = self._stored_fields()
        restored: dict[str, Any] = {}
        for name, annotation in declared.items():
            value = stored[name] if name in stored else getattr(context, name)
            restored[name] = TypeAdapter(annotation).validate_python(value)
        # Validate every field before changing the caller's object.
        for name, value in restored.items():
            setattr(context, name, value)
        self._loaded[id(context)] = (
            context,
            {name: self._revisions.get(name, 0) for name in declared},
        )
        return context

    async def save_context(self, context: object) -> None:
        """Stage all marked fields for the managed commit, rejecting stale local snapshots."""
        self._ensure_open()
        if self._read_only:
            raise ValueError("context writes require an exclusive managed turn")
        declared = _declared_fields(context)
        loaded = self._loaded.get(id(context))
        if loaded is None or loaded[0] is not context:
            raise ValueError("load_context must be called before save_context")
        revisions = loaded[1]
        for name in declared:
            if revisions.get(name) != self._revisions.get(name, 0):
                raise VersionConflict(f"persisted field `{name}` changed since it was loaded")
        updated: dict[str, JsonValue] = dict(self._stored_fields())
        for name, annotation in declared.items():
            try:
                adapter = TypeAdapter(annotation)
                value = adapter.validate_python(getattr(context, name))
                updated[name] = adapter.dump_python(value, mode="json")
            except Exception as exc:
                raise NotSerializable(f"persisted field `{name}`: {exc}") from exc
        state: dict[str, JsonValue] = dict(self._state) if isinstance(self._state, dict) else {}
        state[_STORAGE_KEY] = updated
        # Apply the same JSON validation and size limit as other stored state.
        encoded = dump_state(state, dict[str, JsonValue])
        self._state = copy.deepcopy(encoded)
        self._pending_state = copy.deepcopy(encoded)
        for name in declared:
            self._revisions[name] = self._revisions.get(name, 0) + 1
            revisions[name] = self._revisions[name]

    def _stored_fields(self) -> Mapping[str, JsonValue]:
        return _field_values(self._state)

    def _rebase(self, state: JsonValue) -> None:
        """Refresh a host snapshot without discarding explicitly staged writes."""
        self._ensure_open()
        incoming = copy.deepcopy(state)
        incoming_fields = _field_values(incoming)
        if incoming == self._committed_state:
            return
        if self._pending_state is not None:
            raise VersionConflict("persisted context changed while explicit saves were pending")
        previous_fields = self._stored_fields()
        changed = {
            name
            for name in previous_fields.keys() | incoming_fields.keys()
            if (name in previous_fields) != (name in incoming_fields)
            or previous_fields.get(name) != incoming_fields.get(name)
        }
        self._state = incoming
        self._committed_state = copy.deepcopy(incoming)
        for name in changed:
            self._revisions[name] = self._revisions.get(name, 0) + 1

    def _acknowledge(self, state: JsonValue) -> None:
        """Record the snapshot the host committed, preserving any newer explicit save."""
        self._ensure_open()
        if state is None:
            return
        self._committed_state = copy.deepcopy(state)
        if self._pending_state == state:
            self._pending_state = None

    def close(self) -> None:
        """End this activation's access to the store."""
        self._closed = True
        self._loaded.clear()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("the context store's activation has ended")


def _field_values(state: JsonValue) -> Mapping[str, JsonValue]:
    if state is None:
        return {}
    if not isinstance(state, dict):
        raise ValueError("persisted context state must be an object")
    stored = state.get(_STORAGE_KEY, {})
    if not isinstance(stored, dict):
        raise ValueError("invalid persisted context field record")
    return stored


def _declared_fields(context: object) -> dict[str, Any]:
    if not is_dataclass(context) or isinstance(context, type):
        raise TypeError("context persistence requires a dataclass instance")
    hints = get_type_hints(type(context), include_extras=True)
    return {
        item.name: hints[item.name]
        for item in fields(context)
        if item.metadata.get(_FIELD_MARKER, False)
    }


async def load_context(context: T, *, store: ContextStore) -> T:
    """Restore marked fields using an explicitly supplied store."""
    return await store.load_context(context)


async def save_context(context: object, *, store: ContextStore) -> None:
    """Save marked fields using an explicitly supplied store."""
    await store.save_context(context)
