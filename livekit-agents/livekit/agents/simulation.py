from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, Union

import yaml
from google.protobuf import json_format

from livekit.protocol import agent_simulation as proto

if TYPE_CHECKING:
    from .job import JobContext

# Re-export the generated proto messages as the canonical scenario types.
Scenario = proto.Scenario
ScenarioGroup = proto.ScenarioGroup
SimulationRun = proto.SimulationRun
SimulationDispatch = proto.SimulationDispatch

# Decoded form of a Scenario's `userdata` (arbitrary JSON). On the wire it is a
# JSON-encoded string; in a scenarios.yaml it is written as a nested mapping.
ScenarioUserdata: TypeAlias = dict[str, Union["ScenarioUserdata", Any]]

__all__ = [
    "Scenario",
    "ScenarioGroup",
    "SimulationRun",
    "SimulationDispatch",
    "ScenarioUserdata",
    "SimulationVerdict",
    "SimulationContext",
    "load_scenarios",
    "scenario_group_to_yaml",
]


def _encode_userdata_in_place(scenario: dict[str, Any]) -> None:
    """A scenarios.yaml carries `userdata` as a nested mapping for ergonomics, but
    the proto field is a JSON-encoded string. Encode it before ParseDict."""
    if "userdata" in scenario and not isinstance(scenario["userdata"], str):
        scenario["userdata"] = json.dumps(scenario["userdata"])


def load_scenarios(path: str | Path) -> proto.ScenarioGroup:
    """Load a scenarios.yaml file into a :class:`ScenarioGroup` proto.

    The YAML mirrors the proto field-for-field (``name`` + ``scenarios`` with
    ``label``/``instructions``/``agent_expectations``/``tags``/``userdata``), so
    the conversion is a json_format round-trip. ``userdata`` is written as a
    nested mapping in YAML and JSON-encoded into the proto string field.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"scenarios file {path} must be a mapping with name/scenarios")

    for scenario in raw.get("scenarios", []) or []:
        if isinstance(scenario, dict):
            _encode_userdata_in_place(scenario)

    return json_format.ParseDict(raw, proto.ScenarioGroup())


def scenario_group_to_yaml(group: proto.ScenarioGroup) -> str:
    """Inverse of :func:`load_scenarios` — render a ScenarioGroup as a scenarios.yaml
    string (decoding each scenario's JSON ``userdata`` back into a nested mapping)."""
    data = json_format.MessageToDict(group, preserving_proto_field_name=True)
    for scenario in data.get("scenarios", []):
        if isinstance(scenario.get("userdata"), str) and scenario["userdata"]:
            scenario["userdata"] = json.loads(scenario["userdata"])
    return yaml.safe_dump(data, sort_keys=False)


@dataclass
class SimulationVerdict:
    """A pass/fail verdict for a scenario, with a human-readable reason."""

    success: bool
    reason: str


class SimulationContext:
    """Passed to the ``on_simulation_end`` callback while running under a simulation.

    Carries two verdicts, both recorded for the run:
      - :attr:`simulator_verdict` — the simulator's verdict (its LLM judgment of the chat).
      - :attr:`user_verdict` — your own veto, set via :meth:`fail` from richer checks
        (e.g. comparing mock backend state against the benchmark target in
        ``scenario.userdata``). The effective result is the AND of the two: your check
        can fail a run the simulator passed, but it can never rescue one — so there is
        no ``success()``; not calling :meth:`fail` leaves the simulator's verdict to stand.

    The framework creates and caches this from the simulation room's metadata, so the
    same instance is shared everywhere — read it any time via
    :meth:`JobContext.simulation_context` (``None`` in production). Use :attr:`job_context`
    to reach the running session (``job_context.primary_session``), the room, etc.
    ``simulator_verdict`` and the hydrated ``run`` / ``job`` are filled in right before
    ``on_simulation_end`` is invoked.
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
    def simulation_run_id(self) -> str:
        return self._dispatch.simulation_run_id

    @property
    def job_id(self) -> str:
        return self._dispatch.job_id

    @property
    def scenario(self) -> proto.Scenario:
        return self._scenario

    @property
    def run(self) -> proto.SimulationRun | None:
        return self._run

    @property
    def job(self) -> proto.SimulationRun.Job | None:
        return self._job

    @property
    def simulator_verdict(self) -> SimulationVerdict | None:
        """The simulator's verdict (its LLM judgment of the conversation). Read-only;
        recorded alongside your :attr:`user_verdict`. None until the simulation ends."""
        return self._simulator_verdict

    @property
    def job_context(self) -> JobContext:
        """The :class:`JobContext` for this run — use it to reach the running session
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
