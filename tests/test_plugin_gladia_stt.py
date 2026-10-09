from __future__ import annotations

import pytest

pytestmark = pytest.mark.plugin("gladia")


async def test_update_options_stores_region_for_new_streams():
    from livekit.plugins.gladia import STT

    stt = STT(api_key="test-key", region="eu-west")
    stt.update_options(region="us-west")

    assert stt._opts.region == "us-west"
