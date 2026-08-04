from dataclasses import FrozenInstanceError

import pytest
from agent import CONFIG, AgentConfig

pytestmark = pytest.mark.unit


def test_agent_config_is_the_single_source_of_runtime_identity() -> None:
    assert AgentConfig() == CONFIG
    assert CONFIG.name == "homepage_agent_v3"
    assert CONFIG.tts_voice == "Nate"

    with pytest.raises(FrozenInstanceError):
        CONFIG.tts_voice = "Alex"  # type: ignore[misc]
