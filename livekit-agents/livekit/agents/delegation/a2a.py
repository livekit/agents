"""A delegate reached over A2A on an HTTP endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from ..a2a import A2AClient, TaskInput
from .delegate import Delegate, DelegateStream

if TYPE_CHECKING:
    import httpx


class A2ADelegate(Delegate):
    """An expert served at an HTTP endpoint speaking A2A, ours or not.

    One delegate is one context, so give each session its own::

        AgentSession(llm=realtime_model, delegate=A2ADelegate("http://localhost:8080/fare-desk"))

    A session persisted with ``start(persist=...)`` resumes the context it last had here.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = url
        self._client = A2AClient(url, headers=headers, httpx_client=httpx_client)

    @property
    def client(self) -> A2AClient:
        return self._client

    @property
    def endpoint(self) -> str:
        """The endpoint's name, the last segment of its URL, as its server registered it."""
        return urlsplit(self._url).path.rstrip("/").rsplit("/", 1)[-1]

    def submit(self, task_input: TaskInput) -> DelegateStream:
        return self.client.send(task_input)

    async def aclose(self) -> None:
        await self._client.aclose()
