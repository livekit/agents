from __future__ import annotations

import pytest

from livekit.agents.voice.amd import _fsm as fsm
from livekit.agents.voice.amd.events import AMDCategory as Category

pytestmark = pytest.mark.unit


def test_transition_is_repeatable_and_does_not_mutate_input() -> None:
    state = Category.UNCERTAIN
    event = Category.MACHINE_VM
    first = fsm.transition(state, event)
    assert fsm.transition(state, event) == first
    assert state is Category.UNCERTAIN
    assert first.next_state is Category.MACHINE_VM
    assert first.effects == ()


STAGES = [c for c in Category if c is not Category.WAIT]


@pytest.mark.parametrize("current", STAGES)
@pytest.mark.parametrize("category", list(Category))
def test_stage_transitions(current: Category, category: Category) -> None:
    recommended = {
        Category.UNCERTAIN: set(Category),
        Category.MACHINE_SCREENING: {
            Category.MACHINE_SCREENING,
            Category.HUMAN,
            Category.MACHINE_VM,
            Category.MACHINE_UNAVAILABLE,
            Category.UNCERTAIN,
            Category.WAIT,
        },
        Category.MACHINE_VM: {
            Category.MACHINE_VM,
            Category.HUMAN,
            Category.MACHINE_IVR,
            Category.MACHINE_UNAVAILABLE,
            Category.UNCERTAIN,
            Category.WAIT,
        },
        Category.MACHINE_IVR: {
            Category.MACHINE_IVR,
            Category.HUMAN,
            Category.MACHINE_VM,
            Category.MACHINE_UNAVAILABLE,
            Category.UNCERTAIN,
            Category.WAIT,
        },
        Category.HUMAN: set(),
        Category.MACHINE_UNAVAILABLE: set(),
    }
    state = current
    event = category
    if not recommended[current]:
        with pytest.raises(ValueError, match="invalid AMD transition"):
            fsm.transition(state, event)
        return
    result = fsm.transition(state, event)
    assert result.corrects_stage == (category not in recommended[current])
    if category in {Category.UNCERTAIN, Category.WAIT}:
        assert result.next_state is current
    else:
        assert result.next_state is category
    if category in {Category.HUMAN, Category.MACHINE_UNAVAILABLE}:
        assert result.effects == (fsm.Effect.COMPLETE,)
    elif category is Category.MACHINE_IVR:
        assert result.effects == (fsm.Effect.EXTRACT_MENU,)
    else:
        assert result.effects == ()


@pytest.mark.parametrize(
    ("initial", "corrected"),
    [
        (Category.MACHINE_SCREENING, Category.MACHINE_IVR),
        (Category.MACHINE_VM, Category.MACHINE_SCREENING),
        (Category.MACHINE_IVR, Category.MACHINE_SCREENING),
    ],
)
@pytest.mark.parametrize("bridge", [Category.UNCERTAIN, Category.WAIT])
def test_wait_and_uncertain_keep_the_stage(
    initial: Category, corrected: Category, bridge: Category
) -> None:
    intermediate = fsm.transition(initial, bridge)
    assert intermediate.next_state is initial
    assert intermediate.effects == ()
    assert not intermediate.corrects_stage
    assert fsm.transition(intermediate.next_state, corrected).corrects_stage


def test_same_ivr_state_extracts_each_menu() -> None:
    state = Category.MACHINE_IVR
    result = fsm.transition(state, Category.MACHINE_IVR)
    assert result.next_state == state
    assert result.effects == (fsm.Effect.EXTRACT_MENU,)


@pytest.mark.parametrize(
    ("initial", "corrected"),
    [
        (Category.MACHINE_SCREENING, Category.MACHINE_IVR),
        (Category.MACHINE_VM, Category.MACHINE_SCREENING),
        (Category.MACHINE_IVR, Category.MACHINE_SCREENING),
    ],
)
def test_leaving_the_recommendations_corrects_the_stage(
    initial: Category, corrected: Category
) -> None:
    assert corrected not in fsm.RECOMMENDED[initial]
    result = fsm.transition(initial, corrected)
    assert result.corrects_stage
    assert result.next_state == corrected
    assert result.effects == (
        (fsm.Effect.EXTRACT_MENU,) if corrected == Category.MACHINE_IVR else ()
    )


@pytest.mark.parametrize("category", list(Category))
def test_initial_predictions_are_never_corrections(category: Category) -> None:
    assert not fsm.transition(Category.UNCERTAIN, category).corrects_stage


@pytest.mark.parametrize("stage", [Category.HUMAN, Category.MACHINE_UNAVAILABLE])
def test_terminal_stages_cannot_reopen_detection(stage: Category) -> None:
    with pytest.raises(ValueError, match="invalid AMD transition"):
        fsm.transition(stage, Category.MACHINE_SCREENING)


def test_normal_progression_is_not_a_correction() -> None:
    result = fsm.transition(Category.MACHINE_VM, Category.MACHINE_IVR)
    assert result.next_state is Category.MACHINE_IVR
    assert not result.corrects_stage
