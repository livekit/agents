"""Mock LiveKit Cloud dashboard data for IVR demos."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional


@dataclass(frozen=True)
class CloudAgentRecord:
    name: str
    status: str
    region: str
    modality: str
    active_sessions: int
    uptime_hours: int


@dataclass(frozen=True)
class LatencyMetrics:
    p50_ms: float
    p95_ms: float
    ttft_ms: float


@dataclass(frozen=True)
class CloudAgentSummary:
    total_agents: int
    running: int
    idle: int
    paused: int
    regions: tuple[str, ...]


@dataclass(frozen=True)
class UsageSnapshot:
    llm_cost: float
    tts_cost: float
    stt_cost: float
    balance_remaining: float
    billing_cycle: str
    burn_rate_per_day: float
    last_refreshed: str


@dataclass(frozen=True)
class SipTrunkCounts:
    healthy: int
    degraded: int
    offline: int


@dataclass(frozen=True)
class TelephonyStats:
    inbound_calls: int
    outbound_calls: int
    queued_calls: int
    sip_trunks: SipTrunkCounts
    voicemail_count: int
    avg_handle_time_seconds: int


@dataclass(frozen=True)
class PerformanceOverview:
    llm: LatencyMetrics
    tts: LatencyMetrics
    stt: LatencyMetrics
    last_incidents: tuple[str, ...]


@dataclass(frozen=True)
class SupportOverview:
    open_tickets: int
    sla_tier: str
    pending_callbacks: int
    last_agent_contact: str


@dataclass(frozen=True)
class ProjectRecord:
    name: str
    cloud_agents: tuple[CloudAgentRecord, ...]
    usage: UsageSnapshot
    telephony: TelephonyStats
    performance: PerformanceOverview
    support: SupportOverview


@dataclass(frozen=True)
class AccountRecord:
    label: str
    projects: Mapping[str, ProjectRecord]


class MockLiveKitDashboard:
    """In-memory representation of a subset of LiveKit Cloud data.

    The structure is intentionally rich enough to exercise a multi-layer IVR.
    """

    def __init__(self) -> None:
        self._data: Mapping[str, AccountRecord] = self._load_seed_data()

    def _load_seed_data(self) -> Mapping[str, AccountRecord]:
        horizon_projects = {
            "PRJ-908771": ProjectRecord(
                name="Retail Concierge",
                cloud_agents=(
                    CloudAgentRecord(
                        name="concierge-alpha",
                        status="running",
                        region="us-west",
                        modality="voice",
                        active_sessions=18,
                        uptime_hours=226,
                    ),
                    CloudAgentRecord(
                        name="concierge-fallback",
                        status="idle",
                        region="us-west",
                        modality="voice",
                        active_sessions=0,
                        uptime_hours=72,
                    ),
                    CloudAgentRecord(
                        name="concierge-eu",
                        status="running",
                        region="eu-central",
                        modality="voice",
                        active_sessions=9,
                        uptime_hours=114,
                    ),
                ),
                usage=UsageSnapshot(
                    llm_cost=842.13,
                    tts_cost=412.52,
                    stt_cost=298.72,
                    balance_remaining=4157.49,
                    billing_cycle="2025-10",
                    burn_rate_per_day=92.4,
                    last_refreshed="2025-10-08T23:45:00Z",
                ),
                telephony=TelephonyStats(
                    inbound_calls=1842,
                    outbound_calls=963,
                    queued_calls=7,
                    sip_trunks=SipTrunkCounts(healthy=3, degraded=0, offline=1),
                    voicemail_count=12,
                    avg_handle_time_seconds=276,
                ),
                performance=PerformanceOverview(
                    llm=LatencyMetrics(p50_ms=420, p95_ms=720, ttft_ms=190),
                    tts=LatencyMetrics(p50_ms=210, p95_ms=405, ttft_ms=120),
                    stt=LatencyMetrics(p50_ms=160, p95_ms=295, ttft_ms=90),
                    last_incidents=(
                        "2025-10-04: Elevated TTS latency in us-west",
                        "2025-09-29: LLM token rate limit warnings",
                    ),
                ),
                support=SupportOverview(
                    open_tickets=2,
                    sla_tier="Enterprise",
                    pending_callbacks=1,
                    last_agent_contact="2025-10-08T18:15:00Z",
                ),
            ),
            "PRJ-104552": ProjectRecord(
                name="Logistics Control",
                cloud_agents=(
                    CloudAgentRecord(
                        name="dispatch-core",
                        status="running",
                        region="us-east",
                        modality="voice",
                        active_sessions=25,
                        uptime_hours=512,
                    ),
                    CloudAgentRecord(
                        name="dispatch-analytics",
                        status="running",
                        region="us-east",
                        modality="data-sync",
                        active_sessions=4,
                        uptime_hours=338,
                    ),
                ),
                usage=UsageSnapshot(
                    llm_cost=1325.66,
                    tts_cost=688.14,
                    stt_cost=543.28,
                    balance_remaining=2890.92,
                    billing_cycle="2025-10",
                    burn_rate_per_day=148.2,
                    last_refreshed="2025-10-09T00:05:00Z",
                ),
                telephony=TelephonyStats(
                    inbound_calls=2671,
                    outbound_calls=3145,
                    queued_calls=0,
                    sip_trunks=SipTrunkCounts(healthy=5, degraded=0, offline=0),
                    voicemail_count=4,
                    avg_handle_time_seconds=198,
                ),
                performance=PerformanceOverview(
                    llm=LatencyMetrics(p50_ms=380, p95_ms=640, ttft_ms=170),
                    tts=LatencyMetrics(p50_ms=185, p95_ms=350, ttft_ms=105),
                    stt=LatencyMetrics(p50_ms=140, p95_ms=240, ttft_ms=80),
                    last_incidents=("2025-09-21: SIP carrier maintenance notified",),
                ),
                support=SupportOverview(
                    open_tickets=0,
                    sla_tier="Scale",
                    pending_callbacks=0,
                    last_agent_contact="2025-09-30T16:42:00Z",
                ),
            ),
        }

        northwind_projects = {
            "PRJ-778210": ProjectRecord(
                name="Factory Safety",
                cloud_agents=(
                    CloudAgentRecord(
                        name="safety-monitor",
                        status="running",
                        region="ap-southeast",
                        modality="multimodal",
                        active_sessions=12,
                        uptime_hours=420,
                    ),
                    CloudAgentRecord(
                        name="safety-audit",
                        status="paused",
                        region="us-west",
                        modality="voice",
                        active_sessions=0,
                        uptime_hours=88,
                    ),
                ),
                usage=UsageSnapshot(
                    llm_cost=562.37,
                    tts_cost=221.05,
                    stt_cost=344.91,
                    balance_remaining=612.74,
                    billing_cycle="2025-10",
                    burn_rate_per_day=46.9,
                    last_refreshed="2025-10-08T21:22:00Z",
                ),
                telephony=TelephonyStats(
                    inbound_calls=312,
                    outbound_calls=145,
                    queued_calls=2,
                    sip_trunks=SipTrunkCounts(healthy=2, degraded=1, offline=0),
                    voicemail_count=6,
                    avg_handle_time_seconds=242,
                ),
                performance=PerformanceOverview(
                    llm=LatencyMetrics(p50_ms=455, p95_ms=810, ttft_ms=230),
                    tts=LatencyMetrics(p50_ms=230, p95_ms=420, ttft_ms=140),
                    stt=LatencyMetrics(p50_ms=170, p95_ms=320, ttft_ms=100),
                    last_incidents=(
                        "2025-10-02: Elevated TTFT due to regional congestion",
                        "2025-09-18: Partial outage on stt-ap-southeast",
                    ),
                ),
                support=SupportOverview(
                    open_tickets=1,
                    sla_tier="Ship",
                    pending_callbacks=0,
                    last_agent_contact="2025-10-05T09:55:00Z",
                ),
            )
        }

        accounts = {
            "ACCT-100045": AccountRecord(
                label="Horizon Labs",
                projects=MappingProxyType(horizon_projects),
            ),
            "ACCT-200312": AccountRecord(
                label="Northwind Robotics",
                projects=MappingProxyType(northwind_projects),
            ),
        }

        return MappingProxyType(accounts)

    # -- Account & project validation -------------------------------------------------

    def account_exists(self, account_id: str) -> bool:
        return account_id in self._data

    def get_account_label(self, account_id: str) -> Optional[str]:  # noqa: UP007
        entry = self._data.get(account_id)
        if entry:
            return entry.label
        return None

    def project_exists(self, account_id: str, project_id: str) -> bool:
        return project_id in self._projects_for(account_id)

    def list_projects(self, account_id: str) -> list[str]:
        return list(self._projects_for(account_id).keys())

    def describe_project(self, account_id: str, project_id: str) -> Optional[str]:  # noqa: UP007
        if (projects := self._projects_for(account_id)) and project_id in projects:
            return projects[project_id].name
        return None

    def _projects_for(self, account_id: str) -> Mapping[str, ProjectRecord]:
        account = self._data.get(account_id)
        if account is None:
            return MappingProxyType({})
        return account.projects

    def ensure_account_project(self, account_id: str, project_id: str) -> None:
        if not self.account_exists(account_id):
            raise KeyError(f"Unknown account {account_id}")
        if not self.project_exists(account_id, project_id):
            raise KeyError(f"Unknown project {project_id} for account {account_id}")

    # -- Cloud agents -----------------------------------------------------------------

    def list_cloud_agents(self, account_id: str, project_id: str) -> list[CloudAgentRecord]:
        self.ensure_account_project(account_id, project_id)
        project = self._projects_for(account_id)[project_id]
        return list(project.cloud_agents)

    def get_running_agent_names(self, account_id: str, project_id: str) -> list[str]:
        return [
            a.name for a in self.list_cloud_agents(account_id, project_id) if a.status == "running"
        ]

    def summarize_cloud_agents(self, account_id: str, project_id: str) -> CloudAgentSummary:
        agents = self.list_cloud_agents(account_id, project_id)
        total_running = len([a for a in agents if a.status == "running"])
        return CloudAgentSummary(
            total_agents=len(agents),
            running=total_running,
            idle=len([a for a in agents if a.status == "idle"]),
            paused=len([a for a in agents if a.status == "paused"]),
            regions=tuple(sorted({a.region for a in agents})),
        )

    def find_agent(
        self, account_id: str, project_id: str, agent_name: str
    ) -> Optional[CloudAgentRecord]:  # noqa: UP007
        for agent in self.list_cloud_agents(account_id, project_id):
            if agent.name.lower() == agent_name.lower():
                return agent
        return None

    # -- Usage & billing ---------------------------------------------------------------

    def get_usage_snapshot(self, account_id: str, project_id: str) -> UsageSnapshot:
        self.ensure_account_project(account_id, project_id)
        project = self._projects_for(account_id)[project_id]
        return project.usage

    # -- Telephony ---------------------------------------------------------------------

    def get_telephony_stats(self, account_id: str, project_id: str) -> TelephonyStats:
        self.ensure_account_project(account_id, project_id)
        project = self._projects_for(account_id)[project_id]
        return project.telephony

    # -- Performance -------------------------------------------------------------------

    def get_performance_metrics(self, account_id: str, project_id: str) -> PerformanceOverview:
        self.ensure_account_project(account_id, project_id)
        project = self._projects_for(account_id)[project_id]
        perf = project.performance
        return PerformanceOverview(
            llm=perf.llm,
            tts=perf.tts,
            stt=perf.stt,
            last_incidents=tuple(perf.last_incidents),
        )

    # -- Support -----------------------------------------------------------------------

    def get_support_overview(self, account_id: str, project_id: str) -> SupportOverview:
        self.ensure_account_project(account_id, project_id)
        project = self._projects_for(account_id)[project_id]
        return project.support

    # -- Helpers -----------------------------------------------------------------------

    def aggregate_agent_sessions(self, account_id: str, project_id: str) -> int:
        return sum(
            agent.active_sessions for agent in self.list_cloud_agents(account_id, project_id)
        )

    def aggregate_uptime_hours(self, account_id: str, project_id: str) -> int:
        return sum(agent.uptime_hours for agent in self.list_cloud_agents(account_id, project_id))

    @staticmethod
    def format_latency(metrics: LatencyMetrics) -> str:
        return (
            f"median {metrics.p50_ms:.0f}ms, 95th percentile {metrics.p95_ms:.0f}ms, "
            f"time-to-first-token {metrics.ttft_ms:.0f}ms"
        )


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def format_list(items: Iterable[str]) -> str:
    data = list(items)
    if not data:
        return "none"
    if len(data) == 1:
        return data[0]
    return ", ".join(data[:-1]) + f", and {data[-1]}"
