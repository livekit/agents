"""Host hydration and explicit persistence through generated scope records."""

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from livekit.agents import persisted
from livekit.agents.storage import Storage, VersionConflict
from livekit.agents.storage.session import ContextStore

pytestmark = pytest.mark.unit


class Draft(BaseModel):
    reference: str = ""


@dataclass
class BookingContext:
    service: object
    draft: Draft = persisted(default_factory=Draft)


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
