"""Token authorization at the A2A door: what a credential grants. The order in which the worker
authenticates an HTTP request is pinned by the door's Rust tests."""

import time

import jwt
import pytest

from livekit.agents._generated.storage_class import ThreadInfo
from livekit.agents.a2a import AgentThreadGrant, Credentials, Forbidden, Unauthorized

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
