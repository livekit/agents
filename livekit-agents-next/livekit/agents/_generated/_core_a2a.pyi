# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` star-imports it.

from __future__ import annotations

from .a2a_auth import AuthGrant, AuthOperation, AuthPrincipal

class _core_Credentials:
    def __init__(self, api_key: str, api_secret: str) -> None: ...
    def verify(
        self,
        token: str,
        *,
        agent_name: str = "",
        deployment: str = "",
    ) -> AuthPrincipal: ...
    def principal(
        self,
        authorization: str | None,
        thread_grant: str | None,
        node_authorized: bool,
        agent_name: str,
        deployment: str,
    ) -> AuthPrincipal: ...
    def project_token(self, *, identity: str = "worker", ttl_seconds: int = 3600) -> str: ...
    def thread_grant_token(
        self,
        grant: AuthGrant,
        *,
        identity: str = "",
        ttl_seconds: int = 3600,
    ) -> str: ...

def _core_auth_grant_allows(
    grant: AuthGrant,
    operation: AuthOperation,
    thread_id: str,
    scope: str | None,
) -> bool: ...
