"""`import livekit.agents` stays cheap: every `lk agent dev` launch and hot reload pays it.

Wall-clock limits flake on slow runners, so these budgets count the work an import does, which
is the same on every machine.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

# 826 modules and 18 frozen dataclasses when written; each budget leaves ~10% of headroom.
MODULE_BUDGET = 900
FROZEN_BUDGET = 20
NEVER_IMPORTED = ("unittest", "torch", "transformers", "onnxruntime", "scipy", "pandas")

_PROBE = """
import dataclasses, json, sys
before = set(sys.modules)
import livekit.agents
ours = [m for m in list(sys.modules.values()) if getattr(m, "__name__", "").startswith("livekit.agents")]
frozen = sorted(
    f"{c.__module__}.{c.__qualname__}"
    for m in ours
    for c in vars(m).values()
    if isinstance(c, type)
    and c.__module__ == m.__name__
    and dataclasses.is_dataclass(c)
    and c.__dataclass_params__.frozen
)
print(json.dumps({
    "modules": len(set(sys.modules) - before),
    "frozen": frozen,
    "loaded": [m for m in %r if m in sys.modules],
}))
"""


@pytest.fixture(scope="module")
def imported() -> dict[str, object]:
    # a fresh interpreter: this test process has already imported the package
    out = subprocess.run(
        [sys.executable, "-c", _PROBE % (NEVER_IMPORTED,)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return dict(json.loads(out))


# Catches: a heavy dependency or subsystem the package now imports eagerly.
def test_importing_the_package_loads_a_bounded_number_of_modules(
    imported: dict[str, object],
) -> None:
    assert imported["modules"] <= MODULE_BUDGET


# Catches: records frozen again by default. Each frozen dataclass generates its setattr, delattr,
# hash and pickling hooks at import; a record is frozen only to be hashable (its Rust type
# derives `Hash`) or by an explicit declaration.
def test_only_a_few_dataclasses_are_frozen(imported: dict[str, object]) -> None:
    frozen = imported["frozen"]
    assert isinstance(frozen, list)
    assert len(frozen) <= FROZEN_BUDGET, frozen


# Catches: test tooling or an optional model runtime imported by the package itself.
def test_importing_the_package_loads_no_test_tooling_or_model_runtime(
    imported: dict[str, object],
) -> None:
    assert imported["loaded"] == []
