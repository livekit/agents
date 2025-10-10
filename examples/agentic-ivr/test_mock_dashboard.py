from __future__ import annotations

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mock_dashboard import (
    LatencyMetrics,
    MockLiveKitDashboard,
    PerformanceOverview,
    SupportOverview,
    TelephonyStats,
    UsageSnapshot,
    format_currency,
    format_list,
)


@pytest.fixture()
def dashboard() -> MockLiveKitDashboard:
    return MockLiveKitDashboard()


def test_account_and_project_validation(dashboard: MockLiveKitDashboard) -> None:
    assert dashboard.account_exists("ACCT-100045")
    assert not dashboard.account_exists("ACCT-DOES-NOT-EXIST")

    assert dashboard.project_exists("ACCT-100045", "PRJ-908771")
    assert not dashboard.project_exists("ACCT-100045", "PRJ-NOPE")

    assert dashboard.get_account_label("ACCT-200312") == "Northwind Robotics"
    assert dashboard.describe_project("ACCT-100045", "PRJ-104552") == "Logistics Control"

    with pytest.raises(KeyError):
        dashboard.ensure_account_project("ACCT-100045", "PRJ-NOPE")

    with pytest.raises(KeyError):
        dashboard.ensure_account_project("ACCT-DOES-NOT-EXIST", "PRJ-908771")


def test_list_projects_and_descriptions(dashboard: MockLiveKitDashboard) -> None:
    projects = dashboard.list_projects("ACCT-100045")
    assert projects == ["PRJ-908771", "PRJ-104552"]

    assert dashboard.describe_project("ACCT-100045", "PRJ-908771") == "Retail Concierge"
    assert dashboard.describe_project("ACCT-100045", "PRJ-999999") is None
    assert dashboard.list_projects("ACCT-DOES-NOT-EXIST") == []


def test_cloud_agent_helpers(dashboard: MockLiveKitDashboard) -> None:
    agents = dashboard.list_cloud_agents("ACCT-100045", "PRJ-908771")
    assert [agent.name for agent in agents] == [
        "concierge-alpha",
        "concierge-fallback",
        "concierge-eu",
    ]

    running = dashboard.get_running_agent_names("ACCT-100045", "PRJ-908771")
    assert running == ["concierge-alpha", "concierge-eu"]

    summary = dashboard.summarize_cloud_agents("ACCT-100045", "PRJ-908771")
    assert summary == {
        "total_agents": 3,
        "running": 2,
        "idle": 1,
        "paused": 0,
        "regions": ["eu-central", "us-west"],
    }

    assert dashboard.aggregate_agent_sessions("ACCT-100045", "PRJ-908771") == 27
    assert dashboard.aggregate_uptime_hours("ACCT-100045", "PRJ-908771") == 412

    assert dashboard.aggregate_agent_sessions("ACCT-100045", "PRJ-104552") == 29
    assert dashboard.aggregate_uptime_hours("ACCT-100045", "PRJ-104552") == 850

    found = dashboard.find_agent("ACCT-100045", "PRJ-908771", "CONCIERGE-ALPHA")
    assert found is not None
    assert found.name == "concierge-alpha"

    assert dashboard.find_agent("ACCT-100045", "PRJ-908771", "unknown-agent") is None


def test_usage_snapshot_contents(dashboard: MockLiveKitDashboard) -> None:
    snapshot = dashboard.get_usage_snapshot("ACCT-100045", "PRJ-908771")
    assert isinstance(snapshot, UsageSnapshot)
    assert snapshot.llm_cost == 842.13
    assert snapshot.tts_cost == 412.52
    assert snapshot.stt_cost == 298.72
    assert snapshot.balance_remaining == 4157.49
    assert snapshot.billing_cycle == "2025-10"
    assert snapshot.burn_rate_per_day == 92.4
    assert snapshot.last_refreshed == "2025-10-08T23:45:00Z"


def test_support_and_telephony_are_typed(dashboard: MockLiveKitDashboard) -> None:
    telephony = dashboard.get_telephony_stats("ACCT-100045", "PRJ-908771")
    assert isinstance(telephony, TelephonyStats)
    assert telephony.queued_calls == 7
    assert telephony.sip_trunks.offline == 1
    assert telephony.avg_handle_time_seconds == 276

    support = dashboard.get_support_overview("ACCT-100045", "PRJ-908771")
    assert isinstance(support, SupportOverview)
    assert support.open_tickets == 2
    assert support.sla_tier == "Enterprise"
    assert support.pending_callbacks == 1
    assert support.last_agent_contact == "2025-10-08T18:15:00Z"


def test_performance_overview_contents(dashboard: MockLiveKitDashboard) -> None:
    performance = dashboard.get_performance_metrics("ACCT-100045", "PRJ-908771")
    assert isinstance(performance, PerformanceOverview)
    assert performance.llm == LatencyMetrics(p50_ms=420, p95_ms=720, ttft_ms=190)
    assert performance.tts == LatencyMetrics(p50_ms=210, p95_ms=405, ttft_ms=120)
    assert performance.stt == LatencyMetrics(p50_ms=160, p95_ms=295, ttft_ms=90)
    assert performance.last_incidents == (
        "2025-10-04: Elevated TTS latency in us-west",
        "2025-09-29: LLM token rate limit warnings",
    )


def test_format_helpers() -> None:
    metrics = LatencyMetrics(p50_ms=123.4, p95_ms=456.7, ttft_ms=89.1)
    assert (
        MockLiveKitDashboard.format_latency(metrics)
        == "median 123ms, 95th percentile 457ms, time-to-first-token 89ms"
    )

    assert format_currency(1234.567) == "$1,234.57"
    assert format_list([]) == "none"
    assert format_list(["alpha"]) == "alpha"
    assert format_list(["alpha", "beta", "gamma"]) == "alpha, beta, and gamma"
