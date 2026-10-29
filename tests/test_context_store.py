"""Host hydration and explicit persistence through generated scope records."""

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from livekit.agents import persisted
from livekit.agents.storage import Storage, VersionConflict
from livekit.agents.storage.context_store import ContextStore

pytestmark = pytest.mark.unit


class Draft(BaseModel):
    reference: str = ""


@dataclass
class BookingContext:
    service: object
    draft: Draft = persisted(default_factory=Draft)


async def test_scope_aliases_share_hydrated_values_without_persisting_services() -> None:
    store = ContextStore()
    first = BookingContext(object())
    second = BookingContext(object())
    await store.load_context(first)
    shared_child = store._for_scope("root")
    await shared_child.load_context(second)
    assert first.draft is second.draft
    first.draft.reference = "booking-1"
    assert second.draft.reference == "booking-1"
    await store.save_context(second)
    fresh_child = store._for_scope("booking-child")
    child_app = BookingContext(object())
    await fresh_child.load_context(child_app)
    assert child_app.draft is not first.draft and child_app.draft.reference == ""
    child_app.draft.reference = "child-booking"
    await fresh_child.save_context(child_app)
    records = store._scope_writes
    store._close()

    service = object()
    restored = BookingContext(service)
    next_turn = ContextStore(scopes=[record.scope for record in records])
    await next_turn.load_context(restored)
    assert restored.draft.reference == "booking-1"
    assert restored.service is service
    assert all(set(record.scope.fields) == {"draft"} for record in records)
    resumed_child = next_turn._for_scope("booking-child")
    resumed = BookingContext(object())
    await resumed_child.load_context(resumed)
    assert resumed.draft.reference == "child-booking"
    next_turn._close()


async def test_runtime_supplied_direct_store_saves_before_returning(
    thread_storage: Storage,
) -> None:
    thread = await thread_storage.threads.create_thread(scope="context-test")

    async def open_store() -> ContextStore:
        activation = await thread_storage.threads._core_context_store(thread.thread_id, "desk")
        return ContextStore(scope_id="root:desk", activation=activation)

    first, stale = await open_store(), await open_store()
    app, old = BookingContext(object()), BookingContext(object())
    await first.load_context(app)
    await stale.load_context(old)
    app.draft.reference = "durable"
    await first.save_context(app)
    first._close()

    restored = await open_store()
    loaded = BookingContext(object())
    await restored.load_context(loaded)
    assert loaded.draft.reference == "durable"
    old.draft.reference = "stale"
    with pytest.raises(VersionConflict):
        await stale.save_context(old)
    await stale.load_context(old)
    assert old.draft.reference == "durable"
    restored._close()
    stale._close()
