from __future__ import annotations

from typing import Protocol


class InferenceExecutor(Protocol):
    async def do_inference(self, method: str, data: bytes) -> bytes | None: ...
