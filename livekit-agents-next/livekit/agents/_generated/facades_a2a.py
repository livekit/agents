# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import _bindings, _core

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .a2a_auth import AgentThreadGrant, Principal


class Credentials:
    """`a2a.Credentials`: the worker's project key pair; signing and verification use the LiveKit
    SDK.
    """

    def __init__(self, api_key: str, api_secret: str) -> None:
        self._native = _core._core_Credentials(api_key, api_secret)

    def verify(self, token: str, *, agent_name: str = "", deployment: str = "") -> Principal:
        return self._native.verify(token, agent_name=agent_name, deployment=deployment)

    def _principal(
        self,
        headers: Mapping[str, str],
        *,
        agent_name: str,
        deployment: str = "",
        node_authorized: bool = False,
    ) -> Principal:
        """The caller of a request with these headers (`Credentials.principal`)."""
        return self._native.principal(
            headers,
            agent_name=agent_name,
            deployment=deployment,
            node_authorized=node_authorized,
        )

    def project_token(self, *, identity: str = "worker", ttl_seconds: int = 3600) -> str:
        return self._native.project_token(identity=identity, ttl_seconds=ttl_seconds)

    def thread_grant_token(
        self,
        grant: AgentThreadGrant,
        *,
        identity: str = "",
        ttl_seconds: int = 3600,
    ) -> str:
        """A thread grant for a browser, a device or a delegating agent. It also carries
        `agentEndpoint.call`, so a LiveKit node admits it to an endpoint, which authorizes the
        thread access itself.
        """
        return self._native.thread_grant_token(grant, identity=identity, ttl_seconds=ttl_seconds)


Credentials.__module__ = "livekit.agents.a2a"
