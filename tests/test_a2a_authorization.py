"""Token and HTTP authorization at the A2A door."""

import time
from pathlib import Path

import httpx
import jwt
import pytest
from starlette.requests import HTTPConnection

from livekit.agents import AgentServer
from livekit.agents._generated.storage_class import ThreadInfo
from livekit.agents.a2a import (
    A2ASessionContext,
    AgentThreadGrant,
    Credentials,
    Forbidden,
    Unauthorized,
)
from livekit.agents.storage import SQLiteStorage

pytestmark = pytest.mark.unit

API_KEY = "authorization-test"
API_SECRET = "a-secret-of-at-least-thirty-two-bytes"


@pytest.fixture
def credentials() -> Credentials:
    return Credentials(API_KEY, API_SECRET)


def thread(scope: str = "user:42") -> ThreadInfo:
    return ThreadInfo(
        thread_id="AT_existing",
        scope=scope,
        attributes={},
        created_at=0,
        last_active=0,
        expires_at=0,
        idle_ttl_seconds=0,
        database_id="AT_existing",
    )


def test_thread_grant_requires_each_verb_and_covers_only_its_scope() -> None:
    grant = AgentThreadGrant(
        scope="user:42", create=True, list=True, read=True, write=True, delete=True
    )
    assert grant.can_create() and grant.can_list("user:42")
    for permission in (grant.can_read, grant.can_write, grant.can_delete):
        assert permission(thread())
        assert not permission(thread("user:7"))
    assert not grant.can_create("user:7")
    empty = AgentThreadGrant(all=True)
    assert not empty.can_create() and not empty.can_list("user:42")
    assert not empty.can_read(thread()) and not empty.can_write(thread())
    assert not empty.can_delete(thread())
    assert not AgentThreadGrant(all=True, create=True).can_create("user:42")
    listed = AgentThreadGrant(thread_ids=("AT_existing",), read=True)
    assert listed.can_read(thread("foreign"))
    assert not listed.can_write(thread())


@pytest.mark.parametrize(
    "malformed",
    [
        {"scope": "user:42", "write": "false"},
        {"scope": "user:42", "all": 1},
        {"scope": 42, "write": True},
        {"threadIds": "AT_existing", "read": True},
        {"threadIds": [42], "read": True},
        ["user:42"],
    ],
)
def test_signed_malformed_thread_claims_never_authorize(
    credentials: Credentials,
    malformed: object,
) -> None:
    now = int(time.time())
    token = jwt.encode(
        {
            "iss": API_KEY,
            "sub": "user:42",
            "nbf": now - 5,
            "exp": now + 60,
            "agentThread": malformed,
        },
        API_SECRET,
        algorithm="HS256",
    )
    with pytest.raises(Unauthorized):
        credentials.verify(token)


def test_selected_header_is_the_only_authentication_carrier(credentials: Credentials) -> None:
    token = credentials.thread_grant_token(AgentThreadGrant(scope="user:42", read=True))
    other = credentials.thread_grant_token(AgentThreadGrant(scope="user:7", write=True))
    with pytest.raises(Unauthorized):
        credentials._principal({}, agent_name="agent")
    for invalid in ("Basic unsupported", "Bearer ", ""):
        with pytest.raises(Unauthorized):
            credentials._principal(
                {"Authorization": invalid, "X-LiveKit-Thread-Grant": token},
                agent_name="agent",
            )
    principal = credentials._principal(
        {"Authorization": f"Bearer {token}", "X-LiveKit-Thread-Grant": other},
        agent_name="agent",
    )
    assert principal.grant is not None and principal.grant.scope == "user:42"
    assert (
        credentials._principal({"X-LiveKit-Thread-Grant": token}, agent_name="agent").kind
        == "grant"
    )


def test_endpoint_call_permission_is_independent_of_thread_access(credentials: Credentials) -> None:
    project = credentials.verify(credentials.project_token())
    assert project.endpoint_call and project.grant is None
    grant = credentials.verify(
        credentials.thread_grant_token(AgentThreadGrant(scope="user:42", read=True)),
    )
    assert not grant.endpoint_call and grant.grant is not None
    assert grant.grant.can_read(thread())
    now = int(time.time())
    room_token = jwt.encode(
        {"iss": API_KEY, "nbf": now - 5, "exp": now + 60, "video": {"roomJoin": True}},
        API_SECRET,
        algorithm="HS256",
    )
    with pytest.raises(Forbidden):
        credentials.verify(room_token)


async def test_agent_card_requires_a_token_and_runs_custom_auth_on_every_request(
    credentials: Credentials,
    tmp_path: Path,
) -> None:
    seen: list[str] = []

    def auth(request: HTTPConnection) -> bool:
        seen.append("called")
        return True

    storage = SQLiteStorage(directory=str(tmp_path))
    server = AgentServer(
        storage=storage, agent_name="auth-agent", api_key=API_KEY, api_secret=API_SECRET
    )

    @server.a2a_session(endpoint="auth", description="Checks credentials", auth=auth)
    async def handler(ctx: A2ASessionContext) -> str:
        return "ok"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.http.app), base_url="http://test"
    ) as client:
        path = "/auth/.well-known/agent-card.json"
        refused = await client.get(path)
        assert refused.status_code == 401 and "error" in refused.json()
        query = await client.get(path, params={"access_token": credentials.project_token()})
        assert query.status_code == 401
        allowed = await client.get(
            path, headers={"Authorization": f"Bearer {credentials.project_token()}"}
        )
        assert allowed.status_code == 200
        assert allowed.json()["securityRequirements"]
    assert len(seen) == 3
    await storage.aclose()
