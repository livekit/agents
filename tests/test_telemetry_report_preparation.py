"""Session reports retain complete bounded input and refuse oversized host conversion."""

from __future__ import annotations

import gc
import json
import weakref

import pytest
from scripts.bench.reports import capture_reports, make_report, prepare

from livekit.agents import llm

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.mark.asyncio
async def test_completed_capture_releases_report_samples_before_the_next_workload():
    from livekit.agents.telemetry._export import call_bounded

    async def completed_report():
        with capture_reports() as capture:
            await prepare(make_report([llm.ChatMessage(role="user", content=["hello"])]))
            assert capture.items == {"report-job": 1}
            return weakref.ref(capture)

    gc.collect()
    reference = await completed_report()
    # Drain the native host-reference release queue through the next normal SDK call.
    await call_bounded("report-capture-release", lambda: None, 0, 1)
    assert reference() is None, "a finished capture must not survive until a later full GC pause"


@pytest.mark.asyncio
async def test_ten_thousand_python_items_reach_the_report_and_the_host_sdk():
    items = [llm.ChatMessage(role="user", content=["hello"], id=str(i)) for i in range(10_000)]
    with capture_reports() as capture:
        await prepare(make_report(items))
    assert capture.items == {"report-job": 10_000}
    assert len(json.loads(capture.uploads["report-job"])["items"]) == 10_000


@pytest.mark.asyncio
async def test_oversize_python_item_is_refused_before_native_conversion(monkeypatch):
    from livekit.agents import _core_view

    converted = False

    def conversion_must_not_run(item):
        nonlocal converted
        converted = True
        raise AssertionError("oversize content reached native extraction")

    monkeypatch.setattr(_core_view, "to_core_item", conversion_must_not_run)
    item = llm.ChatMessage(role="user", content=["x" * (1024 * 1024)])
    with capture_reports() as capture, pytest.raises(RuntimeError, match="64 KiB"):
        await prepare(make_report([item]))
    assert not converted
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


@pytest.mark.asyncio
async def test_report_uploader_initializes_its_native_client_off_the_session_loop(monkeypatch):
    import threading

    from livekit.agents import _core

    loop_thread = threading.get_ident()
    constructor_threads: list[int] = []
    with capture_reports(), monkeypatch.context() as patches:
        constructor = _core._core_SessionReportUpload

        def initialize(api_key: str, api_secret: str):
            constructor_threads.append(threading.get_ident())
            return constructor(api_key, api_secret)

        patches.setattr(_core, "_core_SessionReportUpload", initialize)
        await prepare(make_report([]))

    assert len(constructor_threads) == 1
    assert constructor_threads[0] != loop_thread


def test_report_input_rejects_a_single_event_object_instead_of_silently_losing_it() -> None:
    from livekit.agents import _core

    with pytest.raises(TypeError, match="events"):
        _core.session_records(
            _core.UploadGate(),
            {"events": {"type": "agent_state_changed", "new_state": "thinking"}},
            _core.Tagger(),
            lambda item: None,
            capture_content=False,
        )
