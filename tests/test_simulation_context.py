import pytest
from google.protobuf import json_format

from livekit.agents.testing import fake_job_context
from livekit.agents.types import ATTRIBUTE_SIMULATOR, ATTRIBUTE_SIMULATOR_DISPATCH
from livekit.protocol import agent_simulation as sim_pb

pytestmark = pytest.mark.unit


def _dispatch_json() -> str:
    dispatch = sim_pb.SimulationDispatch(
        simulation_run_id="SR_test",
        job_id="SRJ_test",
        scenario=sim_pb.Scenario(label="happy path", instructions="book a slot"),
    )
    return json_format.MessageToJson(dispatch)


def test_simulation_context_resolves_from_job_attributes() -> None:
    # the dispatch attributes ride the job itself, so the context must resolve
    # before any participant joins (no wait_for_participant)
    with fake_job_context() as ctx:
        ctx._info.job.attributes[ATTRIBUTE_SIMULATOR] = "true"
        ctx._info.job.attributes[ATTRIBUTE_SIMULATOR_DISPATCH] = _dispatch_json()

        sim = ctx.simulation_context()
        assert sim is not None
        assert sim._dispatch.simulation_run_id == "SR_test"
        assert sim.scenario.label == "happy path"
        # cached on second call
        assert ctx.simulation_context() is sim
