"""Turn options cross into native execution as generated records."""

import pytest

from livekit.agents import _core
from livekit.agents._generated.storage_options import TurnOptions
from livekit.agents.storage import Storage

from .test_core_turn import Host, thread_of

pytestmark = pytest.mark.unit


async def test_turn_options_apply_scalar_and_channel_policies(thread_storage: Storage) -> None:
    session = await thread_storage.open(await thread_of(thread_storage))
    try:
        all_channels = _core.Turn(
            session,
            "front-desk",
            Host(),
            TurnOptions(policy="interrupt", debounce=0, wait_timeout=1),
        )
        assert all_channels.policy_for("sms") == "interrupt"
        assert all_channels.policy_for("audio") == "interrupt"

        per_channel = _core.Turn(
            session,
            "front-desk",
            Host(),
            TurnOptions(policy={"sms": "batch"}, debounce={"sms": 2}, wait_timeout=1.5),
        )
        assert per_channel.policy_for("sms") == "batch"
        assert per_channel.policy_for("audio") == "interrupt"

        with pytest.raises(TypeError, match="TurnOptions.policy") as invalid:
            _core.Turn(
                session,
                "front-desk",
                Host(),
                TurnOptions(policy="later"),  # type: ignore[arg-type]
            )
        assert isinstance(invalid.value.__cause__, ValueError)
        assert "later" in str(invalid.value.__cause__)
        with pytest.raises(TypeError, match="expected TurnOptions"):
            _core.Turn(
                session,
                "front-desk",
                Host(),
                {"policy": "interrupt"},  # type: ignore[arg-type]
            )
        with pytest.raises((TypeError, ValueError)):
            _core.Turn(session, "front-desk", Host(), TurnOptions(debounce=-1))
    finally:
        session.close()
