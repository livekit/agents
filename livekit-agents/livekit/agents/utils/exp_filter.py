class ExpFilter:
    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._filtered = -1.0

    def reset(self, alpha: float) -> None:
        self._alpha = alpha
        self._filtered = -1.0

    def apply(self, exp: float, sample: float) -> float:
        if self._filtered == -1.0:
            self._filtered = sample
        else:
            a = self._alpha**exp
            self._filtered = a * self._filtered + (1 - a) * sample

        return self._filtered

    def update_base(self, alpha: float) -> None:
        self._alpha = alpha
