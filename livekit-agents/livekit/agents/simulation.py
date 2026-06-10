from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, Union

from livekit.protocol import agent_simulation as proto

if TYPE_CHECKING:
    from .job import JobContext

# Re-export the generated proto messages as the canonical scenario types.
Scenario = proto.Scenario
ScenarioGroup = proto.ScenarioGroup
SimulationRun = proto.SimulationRun
SimulationDispatch = proto.SimulationDispatch
SimulationMode = proto.SimulationMode

# Decoded form of a Scenario's `userdata` (arbitrary JSON). On the wire it is a
# JSON-encoded string; in a scenarios.yaml it is written as a nested mapping.
ScenarioUserdata: TypeAlias = dict[str, Union["ScenarioUserdata", Any]]

__all__ = [
    "Scenario",
    "ScenarioGroup",
    "SimulationRun",
    "SimulationDispatch",
    "SimulationMode",
    "ScenarioUserdata",
    "SimulationVerdict",
    "SimulationContext",
]


@dataclass
class SimulationVerdict:
    """A pass/fail verdict for a scenario, with a human-readable reason."""

    success: bool
    reason: str


class SimulationContext:
    """Passed to the ``on_simulation_end`` callback while running under a simulation.

    Carries two verdicts, both recorded for the run:
      - :attr:`simulator_verdict`: the simulator's verdict (its LLM judgment of the chat).
      - :attr:`user_verdict`: your own veto, set via :meth:`fail` from richer checks
        (e.g. comparing mock backend state against the benchmark target in
        ``scenario.userdata``). The effective result is the AND of the two: your check
        can fail a run the simulator passed, but it can never rescue one, so there is
        no ``success()``; not calling :meth:`fail` leaves the simulator's verdict to stand.

    Use :attr:`job_context` to reach the running session and the room.
    """

    def __init__(self, dispatch: proto.SimulationDispatch, job_ctx: JobContext) -> None:
        self._dispatch = dispatch
        self._scenario = dispatch.scenario
        self._job_ctx = job_ctx
        self._run: proto.SimulationRun | None = None
        self._job: proto.SimulationRun.Job | None = None
        self._simulator_verdict: SimulationVerdict | None = None
        self._user_verdict: SimulationVerdict | None = None

    @property
    def scenario(self) -> proto.Scenario:
        return self._scenario

    @property
    def simulation_mode(self) -> int:
        """How the simulated user interacts with the agent (text chat or audio).
        Unspecified is treated as text, since simulations predating the field
        were all text-only."""
        if self._dispatch.mode == proto.SimulationMode.SIMULATION_MODE_UNSPECIFIED:
            return proto.SimulationMode.SIMULATION_MODE_TEXT
        return self._dispatch.mode

    @property
    def simulation_run(self) -> proto.SimulationRun | None:
        return self._run

    @property
    def simulation_job(self) -> proto.SimulationRun.Job | None:
        return self._job

    @property
    def simulator_verdict(self) -> SimulationVerdict:
        """The simulator's verdict (its LLM judgment of the conversation). Read-only;
        recorded alongside your :attr:`user_verdict`.

        Only available once the simulation has ended, i.e. inside ``on_simulation_end``.
        Raises :class:`RuntimeError` if accessed earlier (e.g. from the entrypoint).
        """
        if self._simulator_verdict is None:
            raise RuntimeError(
                "simulator_verdict is only available inside on_simulation_end "
                "(after the simulation completes)"
            )
        return self._simulator_verdict

    @property
    def job_context(self) -> JobContext:
        """The :class:`JobContext` for this run; use it to reach the running session
        (``job_context.primary_session``), the room, and other job state."""
        return self._job_ctx

    def _begin_finalize(
        self,
        *,
        simulator_verdict: SimulationVerdict,
        run: proto.SimulationRun | None,
        job: proto.SimulationRun.Job | None,
    ) -> None:
        """Internal: populate the simulator verdict / run before on_simulation_end."""
        self._simulator_verdict = simulator_verdict
        self._run = run
        self._job = job

    def userdata(self) -> ScenarioUserdata:
        """The scenario's ``userdata`` decoded from its JSON string (``{}`` if empty)."""
        if not self._scenario.userdata:
            return {}
        data: ScenarioUserdata = json.loads(self._scenario.userdata)
        return data

    def fail(self, reason: str = "") -> None:
        """Veto this run from your own checks (e.g. final DB state diverged).

        The effective result is the AND of both verdicts, so this can only fail a run
        the simulator passed, never rescue one. The simulator's verdict is still
        reported. The last call wins if you call :meth:`fail` more than once.
        """
        self._user_verdict = SimulationVerdict(success=False, reason=reason)

    @property
    def user_verdict(self) -> SimulationVerdict | None:
        """Your veto set via :meth:`fail`, or None if you didn't veto the run."""
        return self._user_verdict
