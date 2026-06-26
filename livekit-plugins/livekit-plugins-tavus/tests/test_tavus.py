import warnings
from unittest.mock import AsyncMock, patch

import pytest

from livekit.agents.utils import http_context
from livekit.plugins.tavus.api import TavusAPI, TavusException
from livekit.plugins.tavus.avatar import AvatarSession


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for v in ("TAVUS_FACE_ID", "TAVUS_PAL_ID", "TAVUS_REPLICA_ID", "TAVUS_PERSONA_ID"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("TAVUS_API_KEY", "test-key")


def _api() -> TavusAPI:
    # session is unused because _post is always mocked in these tests
    return TavusAPI(session=object())  # type: ignore[arg-type]


def _mock_post() -> AsyncMock:
    return AsyncMock(return_value={"conversation_id": "conv1", "persona_id": "pal_auto"})


def _no_deprecation(rec: list[warnings.WarningMessage]) -> bool:
    return not [w for w in rec if issubclass(w.category, DeprecationWarning)]


async def test_new_args_map_to_unchanged_wire_keys():
    api = _api()
    with patch.object(api, "_post", new=_mock_post()) as m:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            cid = await api.create_conversation(face_id="f1", pal_id="p1")
    assert cid == "conv1"
    payload = m.call_args.args[1]
    assert payload["replica_id"] == "f1"
    assert payload["persona_id"] == "p1"
    assert _no_deprecation(rec)


async def test_deprecated_args_still_work_and_warn():
    api = _api()
    with patch.object(api, "_post", new=_mock_post()) as m:
        with pytest.warns(DeprecationWarning) as rec:
            await api.create_conversation(replica_id="r1", persona_id="x1")
    payload = m.call_args.args[1]
    assert payload["replica_id"] == "r1"
    assert payload["persona_id"] == "x1"
    msgs = [str(w.message) for w in rec]
    assert any("replica_id" in s and "face_id" in s for s in msgs)
    assert any("persona_id" in s and "pal_id" in s for s in msgs)


async def test_new_env_vars_fallback(monkeypatch):
    monkeypatch.setenv("TAVUS_FACE_ID", "envf")
    monkeypatch.setenv("TAVUS_PAL_ID", "envp")
    api = _api()
    with patch.object(api, "_post", new=_mock_post()) as m:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            await api.create_conversation()
    payload = m.call_args.args[1]
    assert payload["replica_id"] == "envf"
    assert payload["persona_id"] == "envp"
    assert _no_deprecation(rec)


async def test_deprecated_env_vars_still_work_and_warn(monkeypatch):
    monkeypatch.setenv("TAVUS_REPLICA_ID", "oldf")
    monkeypatch.setenv("TAVUS_PERSONA_ID", "oldp")
    api = _api()
    with patch.object(api, "_post", new=_mock_post()) as m:
        with pytest.warns(DeprecationWarning):
            await api.create_conversation()
    payload = m.call_args.args[1]
    assert payload["replica_id"] == "oldf"
    assert payload["persona_id"] == "oldp"


async def test_missing_face_id_raises():
    api = _api()
    with patch.object(api, "_post", new=_mock_post()):
        with pytest.raises(TavusException, match="TAVUS_FACE_ID must be set"):
            await api.create_conversation(pal_id="p1")


async def test_avatar_session_resolves_new_and_deprecated_args():
    async with http_context.open():
        with pytest.warns(DeprecationWarning):
            deprecated = AvatarSession(replica_id="r9", persona_id="x9")
        assert deprecated._face_id == "r9"
        assert deprecated._pal_id == "x9"

        renamed = AvatarSession(face_id="f9", pal_id="p9")
        assert renamed._face_id == "f9"
        assert renamed._pal_id == "p9"
