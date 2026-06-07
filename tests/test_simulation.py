from __future__ import annotations

import json

import pytest
import yaml

from livekit.agents import (
    ScenarioResult,
    SimulationContext,
    load_scenarios,
    scenario_group_to_yaml,
)
from livekit.protocol import agent_simulation as sim_pb

pytestmark = pytest.mark.unit


_YAML = """\
name: Room booking
scenarios:
  - label: Booking a room for one night
    instructions: Book a room for tmrw night
    agent_expectations: Room booked successfully
    tags:
      feature: room_booking
    userdata:
      target_state:
        booked_rooms: 1
        breakfast: false
"""


def _write(tmp_path, text: str):
    p = tmp_path / "scenarios.yaml"
    p.write_text(text)
    return p


def test_load_scenarios(tmp_path) -> None:
    group = load_scenarios(_write(tmp_path, _YAML))
    assert isinstance(group, sim_pb.ScenarioGroup)
    assert group.name == "Room booking"
    assert len(group.scenarios) == 1

    sc = group.scenarios[0]
    assert sc.label == "Booking a room for one night"
    assert sc.agent_expectations == "Room booked successfully"
    assert dict(sc.tags) == {"feature": "room_booking"}

    # userdata is a JSON-encoded string on the wire
    assert isinstance(sc.userdata, str)
    assert json.loads(sc.userdata)["target_state"]["booked_rooms"] == 1


def test_load_scenarios_round_trip(tmp_path) -> None:
    group = load_scenarios(_write(tmp_path, _YAML))
    rendered = scenario_group_to_yaml(group)
    # round-tripping back through yaml yields the original nested structure
    reparsed = yaml.safe_load(rendered)
    assert reparsed["name"] == "Room booking"
    sc = reparsed["scenarios"][0]
    assert sc["tags"] == {"feature": "room_booking"}
    assert sc["userdata"]["target_state"] == {"booked_rooms": 1, "breakfast": False}


def _dispatch() -> sim_pb.SimulationDispatch:
    sc = sim_pb.Scenario(
        label="s1",
        instructions="do x",
        agent_expectations="x done",
        userdata=json.dumps({"target_state": {"booked_rooms": 2}}),
    )
    return sim_pb.SimulationDispatch(simulation_run_id="run_1", job_id="job_1", scenario=sc)


def test_simulation_context_userdata() -> None:
    ctx = SimulationContext(_dispatch())
    assert ctx.simulation_run_id == "run_1"
    assert ctx.job_id == "job_1"
    assert ctx.scenario.label == "s1"
    assert ctx.userdata()["target_state"]["booked_rooms"] == 2
    # result/run/session are not available until the simulation finalizes
    assert ctx.result is None
    assert ctx.final_result is None


def test_simulation_context_default_verdict() -> None:
    ctx = SimulationContext(_dispatch())
    ctx._begin_finalize(
        result=ScenarioResult(success=True, reason="sim ok"),
        run=None,
        job=None,
        session=None,
    )
    # without an override, the provisional verdict stands
    assert ctx.overridden is False
    assert ctx.final_result == ScenarioResult(success=True, reason="sim ok")


def test_simulation_context_override() -> None:
    ctx = SimulationContext(_dispatch())
    ctx._begin_finalize(
        result=ScenarioResult(success=True, reason="sim ok"),
        run=None,
        job=None,
        session=None,
    )
    ctx.fail(reason="db mismatch")
    assert ctx.overridden is True
    assert ctx.final_result == ScenarioResult(success=False, reason="db mismatch")

    ctx.success()  # reason falls back to the provisional reason
    assert ctx.final_result == ScenarioResult(success=True, reason="sim ok")


def test_empty_userdata() -> None:
    disp = sim_pb.SimulationDispatch(
        simulation_run_id="r", job_id="j", scenario=sim_pb.Scenario(label="s")
    )
    ctx = SimulationContext(disp)
    assert ctx.userdata() == {}
