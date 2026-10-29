"""Session reports retain complete bounded input and refuse oversized host conversion."""

from __future__ import annotations

import json

import pytest

from livekit.agents import llm
from scripts.bench.reports import capture_reports, make_report, prepare

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.mark.asyncio
async def test_ten_thousand_python_items_reach_the_report_and_the_host_sdk():
    items = [llm.ChatMessage(role="user", content=["hello"], id=str(i)) for i in range(10_000)]
    with capture_reports() as capture:
        await prepare(make_report(items))
    assert capture.items == {"report-job": 10_000}
    assert len(json.loads(capture.uploads["report-job"])["items"]) == 10_000


@pytest.mark.asyncio
async def test_oversize_python_item_is_refused():
    item = llm.ChatMessage(role="user", content=["x" * (1024 * 1024)])
    with capture_reports() as capture, pytest.raises(RuntimeError, match="64 KiB"):
        await prepare(make_report([item]))
    assert capture.uploads == {} and capture.items == {}


@pytest.mark.asyncio
async def test_multimodal_history_projects_text_without_copying_media_or_private_caches():
    from livekit import rtc

    image = llm.ImageContent(image=rtc.VideoFrame(2, 2, rtc.VideoBufferType.RGB24, b"\0" * 12))
    image._cache["encoded"] = b"x" * (1024 * 1024)
    audio = llm.AudioContent(
        frame=[rtc.AudioFrame(b"\0" * 960, 24000, 1, 480)], transcript="spoken"
    )
    message = llm.ChatMessage(role="user", content=["hello", image, audio])
    with capture_reports() as capture:
        await prepare(make_report([message]))
    assert capture.items == {"report-job": 1}
    assert json.loads(capture.uploads["report-job"])["items"][0]["content"] == ["hello"]
