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

    One delegate is one conversation, so give each session its own::

        AgentSession(llm=realtime_model, delegate=A2ADelegate("http://localhost:8080/fare-desk"))

    A session persisted with ``start(persist=...)`` resumes the conversation it last had here.
    """

    def __init__(
        self,
        url: str,
        *,
        context_id: str | None = None,
        headers: dict[str, str] | None = None,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = url
        self._context_id = context_id
        self._headers = headers
        self._httpx_client = httpx_client
        # made on the first send, so a resumed session can still say which context it is
        self._client: A2AClient | None = None

    @property
    def client(self) -> A2AClient:
        if self._client is None:
            self._client = A2AClient(
                self._url,
                context_id=self._context_id,
                headers=self._headers,
                httpx_client=self._httpx_client,
            )
        return self._client

    @property
    def context_id(self) -> str | None:
        """The conversation with the endpoint: the one given, else minted on the first send."""
        return self._client.context_id if self._client is not None else self._context_id

    @property
    def endpoint(self) -> str:
        """The endpoint's name, the last segment of its URL, as its server registered it."""
        return urlsplit(self._url).path.rstrip("/").rsplit("/", 1)[-1]

    def resume(self, context_id: str) -> bool:
        if self._client is not None:
            return False
        self._context_id = context_id
        return True

    def submit(self, task_input: TaskInput) -> DelegateStream:
        return self.client.send(task_input)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
