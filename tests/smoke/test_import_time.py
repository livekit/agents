"""`import livekit.agents` stays cheap: every `lk agent dev` launch and hot reload pays it.
The budget counts modules, the same on every machine, not wall-clock time."""

from __future__ import annotations

import json
import subprocess
import sys

# 826 modules when written; the budget leaves ~10% of headroom.
MODULE_BUDGET = 900
NEVER_IMPORTED = ("unittest", "torch", "transformers", "onnxruntime", "scipy", "pandas")

_PROBE = """
import json, sys
before = set(sys.modules)
import livekit.agents
print(json.dumps({
    "modules": len(set(sys.modules) - before),
    "loaded": [m for m in %r if m in sys.modules],
}))
"""


# Catches: a heavy dependency, test tooling or a model runtime the package imports eagerly.
def test_importing_the_package_stays_in_budget_and_loads_no_test_tooling_or_model_runtime() -> None:
    # a fresh interpreter: this test process has already imported the package
    out = subprocess.run(
        [sys.executable, "-c", _PROBE % (NEVER_IMPORTED,)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    imported = json.loads(out)

    assert imported["loaded"] == []
    assert imported["modules"] <= MODULE_BUDGET, imported["modules"]
