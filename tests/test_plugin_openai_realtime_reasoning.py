from __future__ import annotations

from typing import Any

import pytest
from openai.types.realtime import RealtimeReasoning

from livekit.agents.types import APIConnectOptions
from livekit.plugins.openai.realtime.realtime_model import RealtimeModel

pytestmark = pytest.mark.plugin("openai")

_NO_RETRY = APIConnectOptions(max_retry=0, timeout=0.1)


def _session_update_payload(model: RealtimeModel) -> dict[str, Any]:
    sess = model.session()
    try:
        event = sess._create_session_update_event()
        if not isinstance(event, dict):
            event = event.model_dump(by_alias=True, exclude_unset=True, exclude_defaults=False)
        session: dict[str, Any] = event["session"]
        return session
    finally:
        # the model never connects in these tests; just tear the session down
        sess._msg_ch.close()


async def test_reasoning_in_session_update() -> None:
    model = RealtimeModel(
        model="gpt-realtime-2",
        api_key="sk-test",
        reasoning=RealtimeReasoning(effort="low"),
        conn_options=_NO_RETRY,
    )
    try:
        session = _session_update_payload(model)
        assert session["reasoning"] == {"effort": "low"}
    finally:
        await model.aclose()


async def test_reasoning_absent_by_default() -> None:
    model = RealtimeModel(model="gpt-realtime", api_key="sk-test", conn_options=_NO_RETRY)
    try:
        session = _session_update_payload(model)
        assert "reasoning" not in session
    finally:
        await model.aclose()


async def test_reasoning_in_azure_legacy_session_update() -> None:
    # legacy Azure (api_version set) converts to the flat session format
    model = RealtimeModel(
        azure_deployment="gpt-realtime-2",
        api_key="sk-test",
        api_version="2025-08-28",
        base_url="https://example.openai.azure.com/openai",
        reasoning=RealtimeReasoning(effort="medium"),
        conn_options=_NO_RETRY,
    )
    try:
        session = _session_update_payload(model)
        assert session["reasoning"] == {"effort": "medium"}
    finally:
        await model.aclose()


async def test_update_options_changes_reasoning() -> None:
    model = RealtimeModel(
        model="gpt-realtime-2",
        api_key="sk-test",
        reasoning=RealtimeReasoning(effort="low"),
        conn_options=_NO_RETRY,
    )
    try:
        model.update_options(reasoning=RealtimeReasoning(effort="high"))
        assert model._opts.reasoning == RealtimeReasoning(effort="high")

        session = _session_update_payload(model)
        assert session["reasoning"] == {"effort": "high"}
    finally:
        await model.aclose()
