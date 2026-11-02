"""Report SDK preparation hands the remaining operation budget to native upload."""

import asyncio
from types import SimpleNamespace
from typing import cast

import aiohttp
import pytest

from livekit.agents import _core
from livekit.agents.job import SessionReport
from livekit.agents.llm import ChatContext
from livekit.agents.telemetry import traces

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("expired", [False, True])
async def test_report_upload_cannot_restart_an_existing_preparation_deadline(
    monkeypatch: pytest.MonkeyPatch, expired: bool
) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    budgets: list[float] = []

    class Upload:
        async def upload(self, *args: object, timeout: float | None = None) -> bool:
            assert timeout is not None
            budgets.append(timeout)
            return True

    monkeypatch.setattr(_core, "_core_SessionReportUpload", lambda *_: Upload())
    report = SessionReport(
        job_id="budget-job",
        room_id="budget-room",
        room="room",
        options=None,
        events=[],
        chat_history=ChatContext.empty(),
    )
    remaining = -1.0 if expired else 1.0
    deadline = asyncio.get_running_loop().time() + remaining
    async with asyncio.timeout(5):
        operation = traces._upload_session_report(
            agent_name="desk",
            observability_url="http://127.0.0.1:1",
            report=report,
            tagger=_core.Tagger(),
            http_session=cast(aiohttp.ClientSession, SimpleNamespace()),
            deadline=deadline,
        )
        if expired:
            with pytest.raises(TimeoutError, match="preparation deadline"):
                await operation
            assert budgets == []
        else:
            await operation
            assert len(budgets) == 1 and 0 < budgets[0] <= remaining


@pytest.mark.parametrize(
    ("remaining", "message"),
    [
        (0.0, "deadline elapsed"),
        (-1.0, "finite and non-negative"),
        (float("nan"), "finite and non-negative"),
        (float("inf"), "finite and non-negative"),
    ],
)
async def test_native_upload_refuses_expired_or_invalid_budgets_before_url_or_io(
    remaining: float, message: str
) -> None:
    upload = _core._core_SessionReportUpload("key", "secret")
    with pytest.raises(RuntimeError, match=message):
        await upload.upload(
            "not-a-url", "room", "job", False, False, 0, "{}", None, False, timeout=remaining
        )
