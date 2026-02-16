import functools
import inspect
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from ..log import logger
from ..types import NOT_GIVEN
from .misc import is_given


def deprecate_params(
    mapping: dict[str, str],
    *,
    target_version: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Args:
        mapping: {old_param: suggestion}
        target_version: If set, the warning includes "will be removed in {target_version}".

    Example:
    >>> @deprecate_params({
    ...     "old_param": "Use new_param instead",
    ... }, target_version="v2.0")
    ... def my_function(old_param: NotGivenOr[int] = NOT_GIVEN, new_param: int = 0):
    ...     print(old_param)
    >>> my_function(old_param=1)
    WARNING: old_param is deprecated and will be removed in v2.0. Use new_param instead
    1
    >>> my_function(new_param=1) # no warning
    """

    removal = f" and will be removed in {target_version}" if target_version else ""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind_partial(*args, **kwargs)
            by_suggestion: defaultdict[str, list[str]] = defaultdict(list)
            for name, suggestion in mapping.items():
                if is_given(bound.arguments.get(name, NOT_GIVEN)):
                    by_suggestion[suggestion].append(name)

            for suggestion, names in by_suggestion.items():
                params = ", ".join(names)
                logger.warning(
                    f"{params} {'are' if len(names) > 1 else 'is'} deprecated{removal}. {suggestion}",  # noqa: E501
                    stacklevel=2,
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator
