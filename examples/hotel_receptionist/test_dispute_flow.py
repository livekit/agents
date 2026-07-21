from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools_rooms import _needs_offer_first, _say_dispute_offer, _say_dispute_outcome


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


# --- fix 2: no waiver/credit before an offer was actually made ---------------


def test_first_waiver_attempt_is_blocked_and_records_the_offer() -> None:
    offers: set[str] = set()
    assert _needs_offer_first("goodwill_waived", "HTLGH78:late checkout", offers) is True
    assert "HTLGH78:late checkout" in offers


def test_second_attempt_after_offer_goes_through() -> None:
    offers = {"HTLGH78:late checkout"}
    assert _needs_offer_first("goodwill_waived", "HTLGH78:late checkout", offers) is False


def test_credit_offers_are_gated_too() -> None:
    offers: set[str] = set()
    assert _needs_offer_first("credit_offered", "HTLGH78:minibar", offers) is True


def test_non_monetary_outcomes_are_not_gated() -> None:
    offers: set[str] = set()
    for outcome in ("auto_refunded", "explained_no_action", "escalated_to_manager"):
        assert _needs_offer_first(outcome, "HTLGH78:x", offers) is False
    assert not offers


def test_offer_output_presents_policy_and_amount_without_resolving() -> None:
    out = _say_dispute_offer(
        outcome="goodwill_waived",
        refund=4000,
        line_item="late checkout",
        policy_explanation="Late checkout is a standard forty dollar fee per policy.",
    )
    assert "standard forty dollar fee" in out
    assert "40 dollars" in out  # the concrete offer amount
    assert "accepts_offered_resolution" in out  # tells the agent how to proceed
    # nothing here reads like a completed resolution
    assert "Waived" not in out and "case number" not in out.lower()
