"""Labeled model quality, separate from hermetic correctness and live frontend tool choices."""

import os
from dataclasses import asdict
from pathlib import Path

import pytest

from livekit.agents import SessionState
from livekit.agents.inference import LLM

from .framework.models import live_model_environment, write_live_report
from .framework.routing import RoutingCase, RoutingModel, evaluate_routing, routing_cases

pytestmark = pytest.mark.live_models


@pytest.mark.parametrize(
    "case", [pytest.param(case, id=f"{case.split}-{case.id}") for case in routing_cases()]
)
async def test_gemma_recognizes_the_labeled_response(case: RoutingCase, tmp_path: Path) -> None:
    model_id = os.environ.get("E2E_ROUTING_MODEL", "google/gemma-4-31b-it")
    try:
        env = live_model_environment()
        prompt = os.environ.get("E2E_ROUTING_PROMPT")
        async with LLM(
            model_id,
            api_key=env["LIVEKIT_INFERENCE_API_KEY"],
            api_secret=env["LIVEKIT_INFERENCE_API_SECRET"],
            base_url=env["LIVEKIT_INFERENCE_URL"],
        ) as provider:
            model = RoutingModel(provider, prompt=Path(prompt).read_text() if prompt else None)
            await evaluate_routing(case, model, tmp_path)
    except BaseException as error:
        if not (tmp_path / "live-model-report.json").exists():
            # Initialization failures also count. An empty state is explicitly marked below,
            # rather than treating a case without an inference result as if it never ran.
            write_live_report(
                tmp_path,
                models={"frontend": "scripted", "routing": model_id},
                turns=[],
                state=SessionState(),
                observations={
                    "case": asdict(case),
                    "passed": False,
                    "session_observed": False,
                    "error_type": type(error).__name__,
                },
            )
        raise
