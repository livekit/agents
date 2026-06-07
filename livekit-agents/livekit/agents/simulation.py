from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, Union

import yaml
from google.protobuf import json_format

from livekit.protocol import agent_simulation as proto

if TYPE_CHECKING:
    from .voice import AgentSession

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
    "ScenarioResult",
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
class ScenarioResult:
    """The simulator's provisional verdict for a scenario."""

    success: bool
    reason: str


class SimulationContext:
    """Passed to the ``on_simulation_end`` callback while running under a simulation.

    Exposes the scenario, the run, and the simulator's provisional verdict, and lets
    the agent override the outcome based on richer conditions (e.g. comparing mock
    backend state against a benchmark target carried in ``scenario.userdata``).

    The context is created from the simulation room's metadata as soon as the
    entrypoint calls :meth:`JobContext.simulation_context`, so the scenario is
    available immediately. The provisional ``result`` and the hydrated ``run`` /
    ``job`` are filled in later, right before ``on_simulation_end`` is invoked.
    """

    def __init__(self, dispatch: proto.SimulationDispatch) -> None:
        self._dispatch = dispatch
        self._scenario = dispatch.scenario
        self._run: proto.SimulationRun | None = None
        self._job: proto.SimulationRun.Job | None = None
        self._result: ScenarioResult | None = None
        self._session: AgentSession | None = None
        self._final: ScenarioResult | None = None
        self._overridden = False

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
    def result(self) -> ScenarioResult | None:
        """The simulator's provisional verdict (None until the simulation ends)."""
        return self._result

    @property
    def session(self) -> AgentSession | None:
        return self._session

    def _begin_finalize(
        self,
        *,
        result: ScenarioResult,
        run: proto.SimulationRun | None,
        job: proto.SimulationRun.Job | None,
        session: AgentSession | None,
    ) -> None:
        """Internal: populate the verdict/run/session right before on_simulation_end."""
        self._result = result
        self._run = run
        self._job = job
        self._session = session
        self._final = ScenarioResult(success=result.success, reason=result.reason)

    def userdata(self) -> ScenarioUserdata:
        """The scenario's ``userdata`` decoded from its JSON string (``{}`` if empty)."""
        if not self._scenario.userdata:
            return {}
        return json.loads(self._scenario.userdata)

    def _default_reason(self) -> str:
        return self._result.reason if self._result else ""

    def success(self, reason: str | None = None) -> None:
        """Override the outcome: mark this simulation as passed."""
        self._final = ScenarioResult(success=True, reason=reason or self._default_reason())
        self._overridden = True

    def fail(self, reason: str | None = None) -> None:
        """Override the outcome: mark this simulation as failed."""
        self._final = ScenarioResult(success=False, reason=reason or self._default_reason())
        self._overridden = True

    @property
    def final_result(self) -> ScenarioResult | None:
        """The verdict after any override (None until the simulation ends)."""
        return self._final

    @property
    def overridden(self) -> bool:
        return self._overridden
