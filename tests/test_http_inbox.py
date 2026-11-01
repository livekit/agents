"""Ownership invariants for the HTTP server's bounded, cancellation-safe inbox."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

import pytest

from livekit.agents.http import _BoundedInbox

pytestmark = pytest.mark.unit


@dataclass
class Budget:
    capacity: int
    used: int = 0

    def reserve(self, size: int) -> bool:
        assert size >= 0
        if self.used + size > self.capacity:
            return False
        self.used += size
        return True

    def release(self, size: int) -> None:
        assert 0 <= size <= self.used, "capacity must be released exactly once"
        self.used -= size


def inbox(budget: Budget, *, items: int = 4, size: int = 12) -> _BoundedInbox:
    return _BoundedInbox(
        max_items=items, max_bytes=size, reserve=budget.reserve, release=budget.release
    )


def payload_size(payload: str | bytes) -> int:
    return len(payload.encode("utf-8")) if isinstance(payload, str) else len(payload)


async def test_empty_payloads_are_messages_and_still_consume_item_capacity() -> None:
    budget = Budget(0)
    queue = inbox(budget, items=2, size=0)
    assert queue.try_put("") and queue.try_put(b"")
    assert not queue.try_put("")
    assert not queue.try_put("x")
    assert await queue.receive() == ""
    assert await queue.receive() == b""
    queue.close()
    assert await queue.receive() is None
    assert budget.used == queue.queued_bytes == len(queue) == 0


async def test_inboxes_share_capacity_and_close_releases_only_owned_bytes() -> None:
    budget = Budget(5)
    first, second = inbox(budget), inbox(budget)
    assert first.try_put("éé")  # Four bytes, despite containing two characters.
    assert not second.try_put(b"xx")
    assert budget.used == 4 and second.queued_bytes == 0
    assert await first.receive() == "éé"
    assert second.try_put(b"xx")
    assert first.try_put(b"abc")
    assert budget.used == 5
    first.close()
    first.close()
    assert budget.used == second.queued_bytes == 2
    assert first.closed and not first.try_put(b"")
    assert await first.receive() is None
    second.close()
    assert budget.used == 0


async def test_close_wakes_all_receivers_and_close_waiters() -> None:
    budget = Budget(8)
    queue = inbox(budget)
    receivers = [asyncio.create_task(queue.receive()) for _ in range(3)]
    waiters = [asyncio.create_task(queue.wait_closed()) for _ in range(3)]
    await asyncio.sleep(0)
    queue.close()
    assert await asyncio.wait_for(asyncio.gather(*receivers), 1) == [None] * 3
    await asyncio.wait_for(asyncio.gather(*waiters), 1)
    assert budget.used == 0


@pytest.mark.parametrize("items,size", [(-1, 4), (1, -1)])
def test_negative_limits_are_rejected(items: int, size: int) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        inbox(Budget(8), items=items, size=size)


@pytest.mark.parametrize("seed", range(32))
async def test_generated_inbox_sequences_preserve_ownership(seed: int) -> None:
    rng = random.Random(seed)
    budget = Budget(16)
    queues = [inbox(budget), inbox(budget)]
    model: list[list[str | bytes]] = [[], []]
    closed = [False, False]
    initial_tasks = asyncio.all_tasks()
    payloads: list[str | bytes] = ["", b"", "éé", "🙂🙂", b"abc", b"12345678", b"x" * 32]

    def offer(index: int, payload: str | bytes) -> None:
        size = payload_size(payload)
        expected = (
            not closed[index]
            and len(model[index]) < 4
            and sum(map(payload_size, model[index])) + size <= 12
            and sum(payload_size(p) for messages in model for p in messages) + size <= 16
        )
        assert queues[index].try_put(payload) is expected
        if expected:
            model[index].append(payload)

    def delivered(index: int, payload: str | bytes | None) -> None:
        if closed[index]:
            assert payload is None
        else:
            assert payload == model[index].pop(0)

    async def cancel_receive(index: int, *, racing: bool) -> None:
        receiving = asyncio.create_task(queues[index].receive())
        try:
            if racing:
                await asyncio.sleep(0)
                offer(index, rng.choice(payloads))
            for _ in range(rng.randrange(3)):
                await asyncio.sleep(0)
            receiving.cancel()
            try:
                payload = await asyncio.wait_for(receiving, 1)
            except asyncio.CancelledError:
                pass  # A cancelled wait must leave the reference model untouched.
            else:
                delivered(index, payload)
        finally:
            receiving.cancel()
            await asyncio.gather(receiving, return_exceptions=True)

    try:
        for step in range(128):
            index = rng.randrange(2)
            queue = queues[index]
            action = rng.choice(("offer", "receive", "cancel", "race", "close", "wait", "replace"))
            if action == "offer":
                offer(index, rng.choice(payloads))
            elif action == "receive" and (model[index] or closed[index]):
                delivered(index, await asyncio.wait_for(queue.receive(), 1))
            elif action == "cancel":
                await cancel_receive(index, racing=False)
            elif action == "race" and not model[index] and not closed[index]:
                await cancel_receive(index, racing=True)
            elif action in ("close", "replace"):
                queue.close()
                model[index].clear()
                closed[index] = True
                queue.close()  # Repeated close must not release capacity again.
                assert not queue.try_put(b"")
                assert await queue.receive() is None
                if action == "replace":
                    queues[index] = inbox(budget)
                    closed[index] = False
            elif action == "wait":
                waiting = asyncio.create_task(queue.wait_closed())
                try:
                    await asyncio.sleep(0)
                    if closed[index]:
                        await asyncio.wait_for(waiting, 1)
                    else:
                        assert not waiting.done()
                        waiting.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await waiting
                finally:
                    waiting.cancel()
                    await asyncio.gather(waiting, return_exceptions=True)

            context = f"seed={seed}, step={step}, action={action}"
            for current, messages, terminal in zip(queues, model, closed, strict=True):
                assert len(current) == len(messages), context
                assert current.queued_bytes == sum(map(payload_size, messages)), context
                assert current.closed is terminal, context
            assert budget.used == sum(payload_size(p) for messages in model for p in messages), (
                context
            )
            assert 0 <= budget.used <= budget.capacity, context
            assert asyncio.all_tasks() == initial_tasks, context
    finally:
        for queue in queues:
            queue.close()
    assert budget.used == 0
