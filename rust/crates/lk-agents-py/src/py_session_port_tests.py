"""Remote-session callbacks preserve values and raise failures at the host boundary."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from livekit.agents.evals import SimulationContext, SimulationVerdict
from livekit.agents.voice import _session_bindings
from livekit.protocol import agent_simulation as proto

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_run_input_raises_the_original_failure() -> None:
    failure = RuntimeError("input failed")

    async def interrupt(*, force: bool) -> None:
        assert force

    async def run(*, user_input: str) -> None:
        assert user_input == "hello"
        raise failure

    session = SimpleNamespace(interrupt=interrupt, _run=run)
    access = _session_bindings.RemoteSessionAccess(cast(Any, session))
    with pytest.raises(RuntimeError) as raised:
        await access._core_run_input("hello")
    assert raised.value is failure


@pytest.mark.asyncio
async def test_simulation_uses_verdict_records_and_raises_callback_failures(monkeypatch) -> None:
    job = SimpleNamespace()
    context = SimulationContext(proto.SimulationDispatch(), cast(Any, job))
    job.simulation_context = context
    provisional = SimulationVerdict(success=True, reason="simulator passed")

    async def veto(simulation: SimulationContext) -> None:
        assert simulation.simulator_verdict is provisional
        simulation.fail("application veto")

    job._simulation_end_fnc = veto
    monkeypatch.setattr(_session_bindings, "get_job_context", lambda **_: job)
    access = _session_bindings.RemoteSessionAccess(cast(Any, object()))
    verdict = await access._core_finalize_simulation(provisional)
    assert isinstance(verdict, SimulationVerdict)
    assert not verdict.success
    assert verdict.reason == "application veto"

    failure = RuntimeError("simulation failed")

    async def fail(_: SimulationContext) -> None:
        raise failure

    job._simulation_end_fnc = fail
    with pytest.raises(RuntimeError) as raised:
        await access._core_finalize_simulation(provisional)
    assert raised.value is failure
