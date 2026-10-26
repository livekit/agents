"""The scenario file against the grader: every expected key is one it checks, every outbound call
is marked outbound, and there are ten of each."""

from pathlib import Path

import yaml

from pizzeria import Pizzeria
from simulation import grade

SCENARIOS = yaml.safe_load((Path(__file__).parents[1] / "scenarios.yaml").read_text())
KEYS = {
    "placed",
    "cart",
    "cart_includes",
    "not_in_cart",
    "fulfillment",
    "zip",
    "total",
    "tip",
    "name",
    "phone",
    "driver_note",
    "allergies",
    "cancelled",
    "balance",
    "ended_early",
    "unchanged",
}


def test_ten_calls_each_way_with_stable_ids_and_known_keys():
    scenarios = SCENARIOS["scenarios"]
    directions = [s["tags"]["direction"] for s in scenarios]
    assert directions.count("inbound") == 10 and directions.count("outbound") == 10
    assert len({s["id"] for s in scenarios}) == 20
    for s in scenarios:
        userdata = s["userdata"]
        assert set(userdata["expected"]) <= KEYS, s["label"]
        assert (userdata.get("direction") == "outbound") == (
            s["tags"]["direction"] == "outbound"
        ), s["label"]


def test_an_untouched_order_fails_every_outbound_scenario_that_expects_a_change():
    for s in SCENARIOS["scenarios"]:
        if s["tags"]["direction"] != "outbound":
            continue
        expected = s["userdata"]["expected"]
        untouched = grade(
            Pizzeria.out_for_delivery(s["userdata"].get("name")), expected
        )
        assert untouched, f"{s['label']} would pass without the call doing anything"
