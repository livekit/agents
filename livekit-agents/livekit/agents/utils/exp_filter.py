from ..types import NOT_GIVEN, NotGivenOr
from ..utils.misc import is_given


class ExpFilter:
    def __init__(
        self,
        alpha: float,
        max_val: NotGivenOr[float] = NOT_GIVEN,
        min_val: NotGivenOr[float] = NOT_GIVEN,
        initial: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")

        self._alpha = alpha
        self._filtered = initial
        self._max_val = max_val
        self._min_val = min_val

    def reset(
        self,
        alpha: NotGivenOr[float] = NOT_GIVEN,
        initial: NotGivenOr[float] = NOT_GIVEN,
        min_val: NotGivenOr[float] = NOT_GIVEN,
        max_val: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(alpha):
            self._alpha = alpha
        self._filtered = initial
        if is_given(min_val):
            self._min_val = min_val
        if is_given(max_val):
            self._max_val = max_val

    def apply(self, exp: float, sample: NotGivenOr[float] = NOT_GIVEN) -> float:
        if not is_given(sample):
            sample = self._filtered

        if is_given(sample) and not is_given(self._filtered):
            self._filtered = sample
        elif is_given(sample) and is_given(self._filtered):
            a = self._alpha**exp
            self._filtered = a * self._filtered + (1 - a) * sample
        else:
            raise ValueError("sample or initial must be given.")

        if is_given(self._max_val) and self._filtered > self._max_val:
            self._filtered = self._max_val

        if is_given(self._min_val) and self._filtered < self._min_val:
            self._filtered = self._min_val

        return self._filtered

    def filtered(self) -> float:
        return self._filtered if is_given(self._filtered) else -1.0

    @property
    def value(self) -> float | None:
        if not is_given(self._filtered):
            return None
        return self._filtered

    def update_base(self, alpha: float) -> None:
        self._alpha = alpha
