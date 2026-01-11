import functools
import warnings
from collections import defaultdict

from livekit.agents.types import NOT_GIVEN


def deprecate_params(mapping: dict[str, str]):
    """
    Args:
        mapping: {old_param: suggestion}

    Example:
    >>> @deprecate_params({
    ...     "old_param": "Use new_param instead",
    ... })
    ... def my_function(old_param: NotGivenOr[int] = NOT_GIVEN, new_param: int = 0):
    ...     print(old_param)
    >>> my_function(old_param=1)
    DeprecationWarning: old_param is deprecated. Use new_param instead
    1
    >>> my_function(new_param=1)
    >>> print(my_function(new_param=1))
    1
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            by_suggestion: defaultdict[str, list[str]] = defaultdict(list)
            for name, suggestion in mapping.items():
                if kwargs.get(name, NOT_GIVEN) is not NOT_GIVEN:
                    by_suggestion[suggestion].append(name)

            for suggestion, names in by_suggestion.items():
                params = ", ".join(names)
                warnings.warn(
                    f"{params} {'are' if len(names) > 1 else 'is'} deprecated. {suggestion}",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator
