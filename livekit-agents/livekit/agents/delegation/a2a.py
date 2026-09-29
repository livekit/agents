"""A delegate reached over A2A on an HTTP endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..a2a import A2AClient, TaskInput
from .delegate import Delegate, DelegateStream

if TYPE_CHECKING:
    import httpx


class A2ADelegate(Delegate):
    """An expert served at an HTTP endpoint speaking A2A, ours or not.

    One delegate is one conversation, so give each session its own::

        AgentSession(llm=realtime_model, delegate=A2ADelegate("http://localhost:8080/fare-desk"))
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = A2AClient(url, headers=headers, httpx_client=httpx_client)

    @property
    def client(self) -> A2AClient:
        return self._client

    def submit(self, task_input: TaskInput) -> DelegateStream:
        return self._client.send(task_input)

    async def aclose(self) -> None:
        await self._client.aclose()
