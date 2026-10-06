"""Call-category transitions. AMD executes effects and owns the run's lifecycle.

``uncertain`` and ``wait`` are per-turn predictions, not stages. They keep the
current stage. ``RECOMMENDED`` lists the usual next predictions for each stage. It guides
the classifier but does not restrict it: another category corrects an earlier stage.
"""

from dataclasses import dataclass
from enum import Enum, auto

from .events import AMDCategory

# wait does not change the state
RECOMMENDED = {
    AMDCategory.UNCERTAIN: frozenset(AMDCategory),
    AMDCategory.MACHINE_SCREENING: frozenset(
        {
            AMDCategory.MACHINE_SCREENING,
            AMDCategory.HUMAN,
            AMDCategory.MACHINE_VM,
            AMDCategory.MACHINE_UNAVAILABLE,
            AMDCategory.UNCERTAIN,
            AMDCategory.WAIT,
        }
    ),
    AMDCategory.MACHINE_VM: frozenset(
        {
            AMDCategory.MACHINE_VM,
            AMDCategory.HUMAN,
            AMDCategory.MACHINE_IVR,
            AMDCategory.MACHINE_UNAVAILABLE,
            AMDCategory.UNCERTAIN,
            AMDCategory.WAIT,
        }
    ),
    AMDCategory.MACHINE_IVR: frozenset(
        {
            AMDCategory.MACHINE_IVR,
            AMDCategory.HUMAN,
            AMDCategory.MACHINE_VM,
            AMDCategory.MACHINE_UNAVAILABLE,
            AMDCategory.UNCERTAIN,
            AMDCategory.WAIT,
        }
    ),
    AMDCategory.HUMAN: frozenset(),
    AMDCategory.MACHINE_UNAVAILABLE: frozenset(),
}


class Effect(Enum):
    EXTRACT_MENU = auto()
    COMPLETE = auto()


@dataclass(frozen=True)
class Transition:
    next_state: AMDCategory
    effects: tuple[Effect, ...] = ()
    corrects_stage: bool = False
    """Whether the prediction left the recommended transitions."""


def transition(state: AMDCategory, prediction: AMDCategory) -> Transition:
    # terminal stages complete AMD, so nothing follows them
    if not RECOMMENDED[state]:
        raise ValueError(f"invalid AMD transition: {state} -> {prediction}")
    corrects_stage = prediction not in RECOMMENDED[state]
    match prediction:
        case AMDCategory.HUMAN | AMDCategory.MACHINE_UNAVAILABLE:
            return Transition(prediction, (Effect.COMPLETE,), corrects_stage)
        case AMDCategory.MACHINE_IVR:
            return Transition(prediction, (Effect.EXTRACT_MENU,), corrects_stage)
        case AMDCategory.UNCERTAIN | AMDCategory.WAIT:
            return Transition(state)
        case _:
            return Transition(prediction, (), corrects_stage)
