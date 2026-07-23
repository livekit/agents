from __future__ import annotations

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hotel_db import DISPUTE_POLICIES
from tools_rooms import (
    DisputeResolutionStatus,
    RoomToolsMixin,
    _resolve_dispute_outcome,
    _say_dispute_offer,
    _say_dispute_outcome,
    _select_dispute_resolution,
)

from livekit.agents.llm.utils import build_legacy_openai_schema


def _escalated() -> str:
    return _say_dispute_outcome(
        outcome="escalated_to_manager",
        refund=0,
        case_number="DSP-ABC123",
        line_item="no-show charge",
        escalation="manager review",
        policy_explanation=(
            "The booking was card-guaranteed with no cancellation on record, "
            "so the no-show charge stands."
        ),
    )


# --- fix 1: the escalated outcome must carry the policy position -------------


def test_escalated_outcome_includes_policy_explanation() -> None:
    out = _escalated()
    assert "no cancellation on record" in out
    assert "DSP" in out.replace(", ", "").replace(" ", "")  # case number still present


def test_escalated_outcome_instructs_explaining_before_escalation() -> None:
    out = _escalated()
    # the position comes first, the escalation second
    assert out.index("no cancellation on record") < out.index("escalated")


# --- fix 2: pending / accepted / declined are distinct states ----------------


def test_dispute_tool_schema_uses_explicit_resolution_status() -> None:
    schema = build_legacy_openai_schema(RoomToolsMixin().dispute_charge)
    properties = schema["function"]["parameters"]["properties"]
    assert properties["resolution_status"]["enum"] == ["pending", "accepted", "declined"]
    assert "accepts_offered_resolution" not in properties


@pytest.mark.parametrize(
    ("resolution_status", "expected"),
    [
        ("pending", None),
        ("accepted", ("goodwill_waived", 4000)),
        ("declined", ("escalated_to_manager", 0)),
    ],
)
def test_resolution_status_selects_policy_outcome(
    resolution_status: DisputeResolutionStatus,
    expected: tuple[str, int] | None,
) -> None:
    assert (
        _select_dispute_resolution(
            resolution_status=resolution_status,
            accepted_resolution=("goodwill_waived", 4000),
            declined_resolution=("escalated_to_manager", 0),
        )
        == expected
    )


@pytest.mark.parametrize("resolution_status", ["pending", "accepted", "declined"])
def test_automatic_outcome_ignores_resolution_status(
    resolution_status: DisputeResolutionStatus,
) -> None:
    assert _select_dispute_resolution(
        resolution_status=resolution_status,
        accepted_resolution=("auto_refunded", 1800),
        declined_resolution=("auto_refunded", 1800),
    ) == ("auto_refunded", 1800)


def test_late_checkout_acceptance_and_decline_resolve_differently() -> None:
    policy = DISPUTE_POLICIES["late_checkout_fee"]
    assert _resolve_dispute_outcome(
        policy=policy,
        amount_cents=4000,
        line_item_label="Late checkout",
        invoice_line_items=[("Late checkout", 4000)],
        resolution_status="accepted",
    ) == (
        "goodwill_waived",
        4000,
    )
    assert _resolve_dispute_outcome(
        policy=policy,
        amount_cents=4000,
        line_item_label="Late checkout",
        invoice_line_items=[("Late checkout", 4000)],
        resolution_status="declined",
    ) == (
        "escalated_to_manager",
        0,
    )


def test_offer_output_presents_policy_and_amount_without_resolving() -> None:
    out = _say_dispute_offer(
        outcome="goodwill_waived",
        refund=4000,
        line_item="late checkout",
        policy_explanation="Late checkout is a standard forty dollar fee per policy.",
    )
    assert "standard forty dollar fee" in out
    assert "40 dollars" in out  # the concrete offer amount
    assert 'resolution_status="accepted"' in out
    assert 'resolution_status="declined"' in out
    # nothing here reads like a completed resolution
    assert "Waived" not in out and "case number" not in out.lower()


def test_no_refund_policy_asks_if_the_explanation_resolves_the_concern() -> None:
    out = _say_dispute_offer(
        outcome="explained_no_action",
        refund=0,
        line_item="no-show charge",
        policy_explanation="The guaranteed no-show charge stands.",
    )
    assert "ask whether that explanation resolves" in out
    assert 'resolution_status="accepted"' in out
    assert 'resolution_status="declined"' in out
