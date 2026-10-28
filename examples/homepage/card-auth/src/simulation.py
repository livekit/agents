"""Grades a simulated call on what the bank ended up with, against the scenario's `expected`.

Keys, all optional:
    verified: true if the caller must end the call verified, false if they mustn't
    failed_checks: how many identity checks must have failed
    locked_out: true if too many failed checks must have locked them out
    card: active or locked
    case: true if the card must have gone to loss prevention, false if it mustn't
    made_purchase: what the caller must have said about the charge, true or false
    travel_to: where the travel notice must be for
    back_in_days: how many days from today the travel notice must run, or a list of acceptable
        ones
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from livekit.agents import SimulationContext

from bank import Bank
from card_services import BANK_TZ


async def on_simulation_end(sim: SimulationContext) -> None:
    expected = sim.userdata().get("expected")
    if not expected:
        return
    bank: Bank = sim.job_context.primary_session.userdata
    if problems := grade(bank, expected, datetime.now(BANK_TZ).date()):
        sim.fail(reason="; ".join(problems))


def grade(bank: Bank, expected: dict[str, Any], today: date) -> list[str]:
    problems: list[str] = []
    verified = bank.state == "authenticated"
    if "verified" in expected and verified != expected["verified"]:
        problems.append("ended verified" if verified else "never verified the caller")
    failed = sum(a["authenticated"] is False for a in bank.attempts)
    if (want := expected.get("failed_checks")) is not None and failed != want:
        problems.append(f"{failed} failed checks, expected {want}")
    if "locked_out" in expected and bank.locked_out != expected["locked_out"]:
        problems.append(
            "locked the caller out"
            if bank.locked_out
            else "never locked the caller out"
        )

    card = bank.card
    if (state := expected.get("card")) and card.state != state:
        problems.append(f"the card is {card.state}, expected {state}")
    if "case" in expected and (card.case_id is not None) != expected["case"]:
        problems.append(
            "opened a loss prevention case"
            if card.case_id
            else "never opened a loss prevention case"
        )
    if "made_purchase" in expected:
        said = bank.alert.made_by_cardholder if bank.alert else None
        if said != expected["made_purchase"]:
            problems.append(
                f"recorded made_purchase {said}, expected {expected['made_purchase']}"
            )

    if travel_to := expected.get("travel_to"):
        travel = card.travel
        if travel is None or travel_to.lower() not in travel.destination.lower():
            problems.append(
                f"travel notice for {travel.destination if travel else None}, "
                f"expected {travel_to}"
            )
        elif "back_in_days" in expected:
            days = (travel.until - today).days
            allowed = (
                expected["back_in_days"]
                if isinstance(expected["back_in_days"], list)
                else [expected["back_in_days"]]
            )
            if days not in allowed:
                problems.append(
                    f"the travel notice runs {days} days, expected "
                    f"{' or '.join(map(str, allowed))}"
                )
    return problems
