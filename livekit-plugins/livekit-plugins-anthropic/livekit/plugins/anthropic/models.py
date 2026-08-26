from __future__ import annotations

import re
from typing import Literal

# https://docs.anthropic.com/en/docs/about-claude/model-deprecations#model-status

ChatModels = Literal[
    # retired: these ids are no longer served and return 404
    "claude-3-opus-20240229",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-20250219",
    # deprecated, still served
    "claude-3-haiku-20240307",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-opus-4-1-20250805",
    # current
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-opus-5",
    # `claude-fable-5` and `claude-mythos-5` are intentionally left out: thinking cannot
    # be turned off on them and a single request can run for minutes, which does not fit
    # a real-time voice session.
]

ThinkingSupport = Literal["always_on", "configurable", "unknown"]

# Length of the release-date suffix that some model ids carry ("20250514").
_DATE_SEGMENT_LEN = 8

# Separators seen in the wild: "-" everywhere, "." and "/" in gateway prefixes
# ("anthropic.claude-opus-5", "anthropic/claude-opus-5"), "@" in Vertex snapshots
# ("claude-opus-4-5@20251101") and ":" in legacy Bedrock ARNs.
_SEGMENT_SPLIT = re.compile(r"[-_./@:]+")
_LEADING_DIGITS = re.compile(r"^(\d+)")

# The family name anchors the version: modern ids write it before the version
# ("claude-opus-4-6"), the legacy 3.x ids after it ("claude-3-5-sonnet-20241022").
_FAMILIES = frozenset({"opus", "sonnet", "haiku", "fable", "mythos"})

# Assistant prefilling (a trailing assistant message) was removed with the 4.6
# generation: sending one returns a 400. See livekit/agents#4907.
_PREFILL_REMOVED_FROM = (4, 6)

# `temperature`, `top_p` and `top_k` were removed one generation later, with 4.7.
_SAMPLING_REMOVED_FROM = (4, 7)

# From 4.6 onwards every model accepts an explicit `thinking` configuration, so
# thinking can be turned off deterministically instead of relying on the default
# (which is "off" up to 4.8 but "adaptive" on the 5 generation).
_THINKING_CONFIGURABLE_FROM = (4, 6)

# Families whose thinking cannot be turned off: sending `{"type": "disabled"}` returns a
# 400, so the parameter has to be omitted instead. They are not in `ChatModels`, but
# `model` accepts any string, so the guard has to exist at runtime.
_ALWAYS_THINKING_FAMILIES = frozenset({"fable", "mythos"})


def _segment_version(segment: str) -> int | None:
    """Read a version number out of one id segment, or None if it isn't one."""
    if segment.isdigit() and len(segment) == _DATE_SEGMENT_LEN:
        return None  # a release date ("20250514"), not a version

    match = _LEADING_DIGITS.match(segment)
    return int(match.group(1)) if match else None


def _model_version(model: str) -> tuple[int, ...]:
    """Read the version out of an Anthropic model id.

    The version is the run of numeric segments next to the family name: after it on
    current ids (``claude-opus-4-6``, ``claude-sonnet-5``) and before it on the legacy
    3.x ids (``claude-3-5-sonnet-20241022``). Eight-digit segments are release dates,
    not version numbers, and end the run.

    Anchoring on the family name keeps provider-prefixed and suffixed ids working:
    ``anthropic.claude-opus-5`` (Bedrock), ``claude-opus-4-5@20251101`` (Vertex) and a
    proxy alias such as ``gw-1-claude-opus-5`` all resolve to the underlying model.

    Returns an empty tuple for an id that hides the family name; callers then have to
    fall back to the pre-4.6 behaviour rather than guess.
    """
    # A bracketed suffix ("claude-opus-5[1m]") marks a deployment variant, not a version.
    segments = [segment for segment in _SEGMENT_SPLIT.split(model.split("[")[0].lower()) if segment]
    family = next((i for i, segment in enumerate(segments) if segment in _FAMILIES), None)
    if family is None:
        return ()

    after: list[int] = []
    for segment in segments[family + 1 :]:
        version = _segment_version(segment)
        if version is None:
            break
        after.append(version)
    if after:
        return tuple(after)

    before: list[int] = []
    for segment in reversed(segments[:family]):
        version = _segment_version(segment)
        if version is None:
            break
        before.append(version)
    return tuple(reversed(before))


def _model_supports_prefill(model: str) -> bool:
    """Whether the model accepts a prefilled (trailing) assistant message."""
    version = _model_version(model)
    if not version:
        return True

    return version < _PREFILL_REMOVED_FROM


def _model_supports_sampling_params(model: str) -> bool:
    """Whether the model accepts `temperature`, `top_p` and `top_k`."""
    version = _model_version(model)
    if not version:
        return True

    return version < _SAMPLING_REMOVED_FROM


def _model_thinking_support(model: str) -> ThinkingSupport:
    """How the model handles the `thinking` request parameter.

    - ``always_on``: thinking cannot be turned off; sending `{"type": "disabled"}`
      returns a 400, so the parameter must be omitted entirely.
    - ``configurable``: `{"type": "disabled"}` is accepted.
    - ``unknown``: the model predates configurable thinking, or its id could not be
      parsed; the parameter is left out of the request.
    """
    if _ALWAYS_THINKING_FAMILIES.intersection(_SEGMENT_SPLIT.split(model.lower())):
        return "always_on"

    version = _model_version(model)
    if version and version >= _THINKING_CONFIGURABLE_FROM:
        return "configurable"

    return "unknown"
