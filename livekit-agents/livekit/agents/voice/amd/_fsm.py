"""Call-category transitions. AMD executes effects and owns the run's lifecycle.

``uncertain`` and ``wait`` are per-turn predictions, not stages. They keep the
current stage, so the allowed next categories stay constrained by that stage.
"""

from dataclasses import dataclass
from enum import Enum, auto

from .events import AMDCategory

# wait does not change the state
ALLOWED = {
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


def transition(state: AMDCategory, prediction: AMDCategory) -> Transition:
    if prediction not in ALLOWED[state]:
        raise ValueError(f"invalid AMD transition: {state} -> {prediction}")
    match prediction:
        case AMDCategory.HUMAN | AMDCategory.MACHINE_UNAVAILABLE:
            return Transition(prediction, (Effect.COMPLETE,))
        case AMDCategory.MACHINE_IVR:
            return Transition(prediction, (Effect.EXTRACT_MENU,))
        case AMDCategory.UNCERTAIN | AMDCategory.WAIT:
            return Transition(state)
        case _:
            return Transition(prediction)
