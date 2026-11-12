"""Thread metadata is authoritative after updates and deletion by another worker."""

import pytest

from livekit.agents.storage import Storage, ThreadNotFound

pytestmark = pytest.mark.unit


async def test_thread_metadata_refreshes_after_another_worker_updates_or_deletes(
    thread_storages: tuple[Storage, Storage],
) -> None:
    reader, writer = thread_storages
    created = await writer.threads.create_thread()
    before = await reader.threads.get_thread(created.thread_id)
    assert before.attributes == {}
    await writer.threads.update_thread(created.thread_id, attributes={"owner": "new"})
    assert (await reader.threads.get_thread(created.thread_id)).attributes == {"owner": "new"}
    await writer.threads.delete_thread(created.thread_id)
    with pytest.raises(ThreadNotFound):
        await reader.threads.get_thread(created.thread_id)
