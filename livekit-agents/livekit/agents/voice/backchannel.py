from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from livekit import rtc

from ..log import logger
from ..types import NotGivenOr
from ..utils import is_given
from ..utils.audio import audio_frames_from_file
from .background_audio import AudioSource, BuiltinAudioClip, _apply_gain, _frame_gain

if TYPE_CHECKING:
    from .agent_activity import AgentActivity
    from .events import _AgentBackchannelOpportunityEvent

# time-to-first-frame cap for synthesizing a backchannel; a late one is worse than none
_SYNTH_TTFF_TIMEOUT = 0.3
# emit-count window over which a just-emitted clip's "heat" decays back to 0
_HOTNESS_DECAY = 3
_DEFAULT_FREQUENCY = 0.5


@dataclass
class BackchannelConfig:
    """A single backchannel clip and the conditions under which it may play.

    ``source`` is TTS text, an existing audio file path, or an ``AudioSource``.
    ``eot_range`` gates eligibility by the end-of-turn margin: the clip is only
    in the sampling pool when ``lo <= end_of_turn_probability / threshold < hi``.
    Risky lexical words ("yeah") use a low band so they only fire when the user is
    clearly mid-utterance; safe sounds ("mm-hmm") use a high band near the threshold.
    """

    source: str | AudioSource
    probability: float = 1.0
    """Relative weight when eligible (``random.choices`` normalizes across the pool)."""
    eot_range: tuple[float, float] = (0.0, 0.15)
    """Eligible band as fractions of the end-of-turn threshold."""
    volume: float = 1.0
    """Gain applied to the rendered frames before playback."""

    def calculate_weight(self, *, eot_probability: float, eot_threshold: float) -> float:
        if eot_threshold <= 0:
            return 0.0
        frac = eot_probability / eot_threshold
        lo, hi = self.eot_range
        return self.probability if lo <= frac < hi else 0.0


class BackchannelOptions(TypedDict, total=False):
    """Agent backchannels — short acknowledgments emitted while the user is still
    speaking. Nested under ``ExpressiveOptions["backchannel"]``."""

    frequency: float
    """Independent pre-gate in ``[0, 1]``: chance of even attempting a backchannel
    on a given opportunity."""
    source: NotGivenOr[list[str | AudioSource | BackchannelConfig]]


# Two tiers: safe sounds near the threshold, risky words only at very low eot.
DEFAULT_BACKCHANNEL_SOURCE: list[str | AudioSource | BackchannelConfig] = [
    BackchannelConfig("mm-hmm", eot_range=(0.15, 1.0)),
    BackchannelConfig("uh-huh", eot_range=(0.15, 1.0)),
    BackchannelConfig("hmm", eot_range=(0.15, 1.0)),
    BackchannelConfig("okay", eot_range=(0.0, 0.15)),
    BackchannelConfig("yeah", eot_range=(0.0, 0.15)),
    BackchannelConfig("right", eot_range=(0.0, 0.15)),
    BackchannelConfig("i see", eot_range=(0.0, 0.15)),
]

DEFAULT_BACKCHANNEL_OPTIONS: BackchannelOptions = {
    "frequency": _DEFAULT_FREQUENCY,
    "source": DEFAULT_BACKCHANNEL_SOURCE,
}


def resolve_backchannel_options(
    backchannel: NotGivenOr[
        bool | list[str | AudioSource | BackchannelConfig] | BackchannelOptions
    ],
) -> BackchannelOptions | None:
    """Normalize the ``ExpressiveOptions["backchannel"]`` value.

    ``NOT_GIVEN``/``True`` → defaults; ``False`` → ``None`` (disabled); a list →
    ``source`` sugar over the defaults; a dict → merged over the defaults.
    """
    if not is_given(backchannel) or backchannel is True:
        return cast("BackchannelOptions", dict(DEFAULT_BACKCHANNEL_OPTIONS))
    if backchannel is False:
        return None
    if isinstance(backchannel, list):
        return cast("BackchannelOptions", {**DEFAULT_BACKCHANNEL_OPTIONS, "source": backchannel})
    overrides = cast("BackchannelOptions", backchannel)
    return cast("BackchannelOptions", {**DEFAULT_BACKCHANNEL_OPTIONS, **overrides})


def _as_config(entry: str | AudioSource | BackchannelConfig) -> BackchannelConfig:
    if isinstance(entry, BackchannelConfig):
        return entry
    # bare str / AudioSource: always eligible, default relative weight
    return BackchannelConfig(source=entry, eot_range=(0.0, 1.0))


def _is_text(source: str | AudioSource) -> bool:
    """A ``str`` is TTS text unless it points at an existing file."""
    return isinstance(source, str) and not Path(source).is_file()


def _clip_key(source: str | AudioSource) -> str:
    if isinstance(source, BuiltinAudioClip):
        return f"builtin:{source.value}"
    if isinstance(source, str):
        return source
    return f"iter:{id(source)}"


def _with_volume(frame: rtc.AudioFrame, volume: float) -> rtc.AudioFrame:
    gain = _frame_gain(0, frame.samples_per_channel, None, 0.0, 0.0, frame.sample_rate, volume)
    return _apply_gain(frame, gain)


async def _iter_frames(frames: list[rtc.AudioFrame]) -> AsyncGenerator[rtc.AudioFrame, None]:
    for f in frames:
        yield f


def _decode_source(source: AudioSource) -> AsyncIterator[rtc.AudioFrame]:
    if isinstance(source, BuiltinAudioClip):
        return audio_frames_from_file(source.path())
    if isinstance(source, str):
        return audio_frames_from_file(source)
    return source  # already an AsyncIterator[rtc.AudioFrame]


class _BackchannelEmitter:
    """Per-activity backchannel player.

    On each opportunity it pre-gates by ``frequency``, builds the eligible pool via
    each config's ``calculate_weight`` (EOT band) and an emit-count anti-repeat
    decay, then picks one with ``random.choices``. Clips are rendered once into a
    frame cache (text via the activity's TTS under a 300ms time-to-first-frame cap,
    files/sources via the audio decoder) and played through ``activity.say`` —
    uninterruptible and excluded from chat history. The first use of a clip is
    rendered on the fly, cached for instant replay, and emitted as soon as it is
    ready; if rendering misses the time-to-first-frame budget nothing is emitted
    that round and the clip is retried later.
    """

    def __init__(self, options: BackchannelOptions) -> None:
        self._frequency = options.get("frequency", _DEFAULT_FREQUENCY)
        source = options.get("source")
        if not is_given(source) or source is None:
            source = DEFAULT_BACKCHANNEL_SOURCE
        self._configs = [_as_config(s) for s in source]

        self._cache: dict[str, list[rtc.AudioFrame]] = {}
        self._pending: set[str] = set()
        self._emit_counter = 0
        self._last_emit_index: dict[str, int] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def maybe_emit(self, ev: _AgentBackchannelOpportunityEvent, activity: AgentActivity) -> None:
        if not self._configs:
            return

        roll = random.random()
        if roll >= self._frequency:
            logger.debug(
                "backchannel skipped: frequency gate",
                extra={"frequency": self._frequency, "roll": roll},
            )
            return

        eot_fraction = (
            ev.end_of_turn_probability / ev.end_of_turn_threshold
            if ev.end_of_turn_threshold > 0
            else None
        )

        pool: list[BackchannelConfig] = []
        weights: list[float] = []
        for cfg in self._configs:
            w = cfg.calculate_weight(
                eot_probability=ev.end_of_turn_probability,
                eot_threshold=ev.end_of_turn_threshold,
            )
            w *= self._cooldown(_clip_key(cfg.source))
            if w > 0:
                pool.append(cfg)
                weights.append(w)

        if not pool:
            logger.debug(
                "backchannel skipped: no clip eligible at this end-of-turn fraction",
                extra={
                    "eot_probability": ev.end_of_turn_probability,
                    "eot_threshold": ev.end_of_turn_threshold,
                    "eot_fraction": eot_fraction,
                },
            )
            return

        cfg = random.choices(pool, weights=weights, k=1)[0]
        key = _clip_key(cfg.source)
        frames = self._cache.get(key)
        if frames is None:
            logger.debug(
                "backchannel rendering then emitting (first use of clip)",
                extra={"clip": key, "eot_fraction": eot_fraction},
            )
            self._render_and_play(cfg, key, activity)
            return

        logger.debug(
            "backchannel emitting",
            extra={"clip": key, "eot_fraction": eot_fraction, "pool_size": len(pool)},
        )
        self._play(cfg, key, frames, activity)

    def _cooldown(self, key: str) -> float:
        last = self._last_emit_index.get(key)
        if last is None:
            return 1.0
        heat = max(0.0, 1.0 - (self._emit_counter - last) / _HOTNESS_DECAY)
        return 1.0 - heat

    def _play(
        self,
        cfg: BackchannelConfig,
        key: str,
        frames: list[rtc.AudioFrame],
        activity: AgentActivity,
    ) -> None:
        if cfg.volume != 1.0:
            frames = [_with_volume(f, cfg.volume) for f in frames]

        source = cfg.source
        transcript = source if isinstance(source, str) and not Path(source).is_file() else ""
        try:
            activity.say(
                transcript,
                audio=_iter_frames(frames),
                allow_interruptions=False,
                add_to_chat_ctx=False,
            )
        except RuntimeError:
            # scheduling paused / activity closing — drop the backchannel silently
            return

        self._emit_counter += 1
        self._last_emit_index[key] = self._emit_counter

    def _render_and_play(self, cfg: BackchannelConfig, key: str, activity: AgentActivity) -> None:
        if key in self._pending:
            return
        self._pending.add(key)
        task = asyncio.create_task(self._render(cfg, key, activity))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(lambda _: self._pending.discard(key))

    async def _render(self, cfg: BackchannelConfig, key: str, activity: AgentActivity) -> None:
        try:
            if _is_text(cfg.source):
                frames = await self._synthesize(str(cfg.source), activity)
            else:
                frames = [f async for f in _decode_source(cfg.source)]
        except Exception:
            logger.exception("backchannel: failed to render clip")
            return

        # nothing usable (e.g. TTS missed the time-to-first-frame budget): leave the
        # clip uncached so a later opportunity can retry, and emit nothing this time
        if not frames:
            return

        self._cache[key] = frames
        self._play(cfg, key, frames, activity)  # play now that it's rendered

    async def _synthesize(self, text: str, activity: AgentActivity) -> list[rtc.AudioFrame]:
        if activity.tts is None:
            return []

        frames: list[rtc.AudioFrame] = []
        stream = activity.tts.synthesize(text)
        try:
            it = stream.__aiter__()
            first = await asyncio.wait_for(it.__anext__(), timeout=_SYNTH_TTFF_TIMEOUT)
            frames.append(first.frame)
            async for ev in it:
                frames.append(ev.frame)
        except asyncio.TimeoutError:
            logger.debug("backchannel: TTS exceeded time-to-first-frame budget, skipping clip")
            return []
        except StopAsyncIteration:
            pass
        finally:
            await stream.aclose()

        return frames
