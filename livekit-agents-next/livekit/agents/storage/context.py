from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import _MISSING_TYPE, MISSING, dataclass, field, fields, is_dataclass
from typing import (
    Protocol,
    Self,
    TypeAlias,
    TypeVar,
    get_args,
    get_type_hints,
    overload,
    runtime_checkable,
)

from pydantic import JsonValue, TypeAdapter

from .._core import _core_ContextActivation
from .._generated import facades_storage_context as _context_facades
from .._generated.facades_storage import Transaction
from .._generated.storage_context import AppStatement, ContextField, ContextScope, ContextScopeWrite
from .._generated.storage_errors import NotSerializable

T = TypeVar("T")
_FIELD_MARKER = "livekit.persisted"


@overload
def persisted(*, default_factory: Callable[[], T]) -> T: ...


@overload
def persisted(*, default: T) -> T: ...


def persisted(
    *,
    default: object = MISSING,
    default_factory: Callable[[], object] | _MISSING_TYPE = MISSING,
) -> object:
    """Mark a dataclass field for explicit context loading and saving."""
    metadata = {_FIELD_MARKER: True}
    if default_factory is MISSING:
        return field(default=default, metadata=metadata)
    if default is not MISSING:
        raise ValueError("cannot specify both default and default_factory")
    return field(default_factory=default_factory, metadata=metadata)


@dataclass
class _LoadedContext:
    context: object
    revision: int
    names: set[str]


@dataclass
class _HydratedScope:
    loaded: dict[int, _LoadedContext] = field(default_factory=dict)
    values: dict[str, object] = field(default_factory=dict)
    revision: int | None = None


SqlValue: TypeAlias = str | int | float | bool | None
"""A bound parameter; SQLite stores a boolean as ``0`` or ``1``."""

Row: TypeAlias = list[SqlValue]
"""One result row: its column values in select order."""


class ContextStore(_context_facades.ContextStore):
    """``ctx.store``: loads and saves persisted context fields and opens transactions."""

    def __init__(
        self,
        state: object | None = None,
        *,
        read_only: bool = False,
        scopes: Sequence[ContextScope] = (),
        scope_id: str = "root",
        activation: _core_ContextActivation | None = None,
    ) -> None:
        self._activation = activation or _core_ContextActivation(list(scopes), state, read_only)
        self._scope_id = scope_id
        self._hydration: dict[str, _HydratedScope] = {scope_id: _HydratedScope()}
        self._objects = self._hydration[scope_id]
        self._owns_activation = True
        self._disposed = False

    def _for_scope(self, scope_id: str) -> ContextStore:
        self._ensure_open()
        view = ContextStore(scope_id=scope_id, activation=self._activation)
        view._hydration = self._hydration
        view._objects = self._hydration.setdefault(scope_id, _HydratedScope())
        view._owns_activation = False
        return view

    async def load_context(self, context: T) -> T:
        """Restore ``context``'s persisted fields, checked against their saved schemas."""
        self._ensure_open()
        adapters = _field_adapters(context)
        declarations = [
            ContextField(name=name, schema=adapter.json_schema(), value=None)
            for name, adapter in adapters.items()
        ]
        values = await self._activation.load(self._scope_id, _identity(context), declarations)
        restored: dict[str, object] = {}
        for name, adapter in adapters.items():
            if values.revision == self._objects.revision and name in self._objects.values:
                restored[name] = self._objects.values[name]
            else:
                value = values.fields[name] if name in values.fields else getattr(context, name)
                restored[name] = adapter.validate_python(value)
        # A failed validator must not leave a partially hydrated application object.
        self._objects.loaded[id(context)] = _LoadedContext(context, values.revision, set(adapters))
        self._publish(restored, values.revision, restored=True)
        return context

    async def save_context(self, context: object) -> None:
        """Validate and save ``context``'s persisted fields; ``load_context`` must run first."""
        self._ensure_open()
        loaded = self._objects.loaded.get(id(context))
        if loaded is None or loaded.context is not context:
            raise ValueError("load_context must be called before save_context")
        declarations: list[ContextField] = []
        validated: dict[str, object] = {}
        for name, adapter in _field_adapters(context).items():
            try:
                value = adapter.validate_python(getattr(context, name))
                encoded = adapter.dump_python(value, mode="json")
                declarations.append(
                    ContextField(name=name, schema=adapter.json_schema(), value=encoded)
                )
                validated[name] = value
            except Exception as exc:
                raise NotSerializable(f"persisted field `{name}`: {exc}") from exc
        values = await self._activation.save(
            self._scope_id, _identity(context), loaded.revision, declarations
        )
        self._publish(validated, values.revision)

    def transaction(self) -> AbstractAsyncContextManager[Transaction]:
        """A transaction whose writes commit or roll back with the managed turn, like a nested
        savepoint; outside a managed turn it commits when the block exits."""
        return self._transaction(independent=False)

    def independent_transaction(self) -> AbstractAsyncContextManager[Transaction]:
        """A transaction whose writes commit when the block exits, whatever the turn does
        next."""
        return self._transaction(independent=True)

    def _publish(self, values: dict[str, object], revision: int, *, restored: bool = False) -> None:
        if revision != self._objects.revision:
            self._objects.values.clear()
        self._objects.values.update(values)
        self._objects.revision = revision
        for loaded in self._objects.loaded.values():
            if not restored or loaded.names <= values.keys():
                loaded.revision = revision
            for name in loaded.names & values.keys():
                setattr(loaded.context, name, values[name])

    @property
    def _scope_writes(self) -> list[ContextScopeWrite]:
        return self._activation.writes

    @property
    def _statements(self) -> list[AppStatement]:
        return self._activation.statements

    @property
    def _closed(self) -> bool:
        return self._disposed or self._activation.closed

    def _core_release_hydration(self) -> None:
        self._disposed = True
        for scope in self._hydration.values():
            scope.loaded.clear()
            scope.values.clear()

    def _close(self) -> None:
        self._disposed = True
        if self._owns_activation:
            self._activation.close()
            self._core_release_hydration()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("the context store's activation has ended")


def _identity(context: object) -> str:
    return f"{type(context).__module__}.{type(context).__qualname__}"


def _field_adapters(context: object) -> dict[str, TypeAdapter[object] | _RecordAdapter]:
    if not is_dataclass(context) or isinstance(context, type):
        raise TypeError("context persistence requires a dataclass instance")
    hints = get_type_hints(type(context), include_extras=True)
    return {
        item.name: _field_adapter(hints[item.name])
        for item in fields(context)
        if item.metadata.get(_FIELD_MARKER, False)
    }


async def load_context(context: T, *, store: ContextStore) -> T:
    """Restore marked fields using an explicitly supplied store."""
    return await store.load_context(context)


async def save_context(context: object, *, store: ContextStore) -> None:
    """Save marked fields using an explicitly supplied store."""
    await store.save_context(context)


@runtime_checkable
class _PersistedRecord(Protocol):
    def _to_json(self) -> str: ...

    @classmethod
    def _from_json(cls, data: str) -> Self: ...


@dataclass(frozen=True)
class _RecordAdapter:
    record: type[_PersistedRecord]
    optional: bool

    def json_schema(self) -> dict[str, JsonValue]:
        schema: dict[str, JsonValue] = {
            "type": "string",
            "format": f"livekit.record:{self.record.__module__}.{self.record.__qualname__}",
        }
        return {"anyOf": [schema, {"type": "null"}]} if self.optional else schema

    def validate_python(self, value: object) -> _PersistedRecord | None:
        if value is None and self.optional:
            return None
        if isinstance(value, self.record):
            return value
        if isinstance(value, str):
            return self.record._from_json(value)
        raise TypeError(f"expected {self.record.__qualname__} or its saved JSON")

    def dump_python(self, value: object, *, mode: str) -> JsonValue:
        record = self.validate_python(value)
        return None if record is None else record._to_json()


def _field_adapter(annotation: object) -> TypeAdapter[object] | _RecordAdapter:
    candidates = get_args(annotation) or (annotation,)
    records = [
        item for item in candidates if isinstance(item, type) and issubclass(item, _PersistedRecord)
    ]
    if records:
        if len(records) != 1 or any(item not in (records[0], type(None)) for item in candidates):
            raise TypeError("a persisted record field must name one record, optionally None")
        return _RecordAdapter(records[0], type(None) in candidates)
    return TypeAdapter(annotation)
