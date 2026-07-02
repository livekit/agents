import os

from ..types import NotGivenOr
from .misc import is_given


def resolve_env_var(val: NotGivenOr[str], *env_vars: str, default: str = "") -> str:
    """
    Resolve an environment variable from a list of potential sources.

    Args:
        val: The value to resolve.
        *env_vars: The environment variables to check. Order matters, the first non-None value will be returned.
        default: The default value to return if no environment variables are set.

    Returns:
        The resolved environment variable.

    Examples:
    >>> resolve_env_var(
    ...     NOT_GIVEN,
    ...     "ABC_URL",
    ...     default="https://agent-gateway.livekit.cloud/v1",
    ... )
    "https://agent-gateway.livekit.cloud/v1"
    """
    if is_given(val):
        return val
    for env_var in env_vars:
        curr_val = os.getenv(env_var, None)
        if curr_val is not None and curr_val != "":
            return curr_val
    return default
