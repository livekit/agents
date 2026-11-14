//! The test that every facade object is freed once closed: one case per `#[host(facade)]`
//! class, each run by the handwritten scenario of its public path in `facade_scenarios.py`.

use crate::decl::Declarations;

pub const PATH: &str = "tests/host/test_facades_freed.py";

pub fn emit(declared: &Declarations) -> String {
    // Each public path, and the generated class: the public one or its base.
    let mut classes: Vec<(String, String)> = declared
        .facades
        .iter()
        .map(|f| (format!("{}.{}", f.path, f.name), format!("{}.{}", f.module, f.name)))
        .collect();
    classes.sort();
    classes.dedup();
    let cases: String = classes.iter().map(|(p, g)| format!("    {p:?}: {g},\n")).collect();
    let mut modules: Vec<&str> = declared.facades.iter().map(|f| f.module.as_str()).collect();
    modules.sort();
    modules.dedup();
    let imports = modules.join(",\n    ");
    format!(
        r#"# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi: `cargo run -p lk-agents-codegen -- --write`.
"""Each facade object is freed once used, closed and dropped: the core holds no strong
reference to a host object that reaches it, a cycle the collector cannot see."""

from __future__ import annotations

import gc
import weakref

import pytest

from livekit.agents._generated import (
    {imports},
)
from tests.fakes import until

from .facade_scenarios import SCENARIOS

FACADES: dict[str, type] = {{
{cases}}}


def test_every_facade_has_a_scenario() -> None:
    assert sorted(SCENARIOS) == sorted(FACADES)


@pytest.mark.parametrize("facade", sorted(FACADES))
async def test_a_closed_facade_object_is_freed(facade: str) -> None:
    closed = await SCENARIOS[facade]()
    assert isinstance(closed, FACADES[facade])
    freed = weakref.ref(closed)
    del closed
    await until(lambda: gc.collect() >= 0 and freed() is None, f"a closed {{facade}} is freed")
"#
    )
}
