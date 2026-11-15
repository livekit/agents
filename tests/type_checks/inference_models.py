"""Type checks for the hosted inference model and option vocabularies, run by
``.github/check_types.py``.

Each ``type: ignore[code]`` marks an error the type checker must report; strict mode reports an
ignore that is no longer needed, so the file fails both when a check is lost and when a new
error appears.
"""

from livekit.agents.inference.stt import SpeechmaticsModels, SpeechmaticsOptions


def speechmatics_models() -> list[SpeechmaticsModels]:
    retired: SpeechmaticsModels = "speechmatics/enhanced"  # type: ignore[assignment]
    return ["speechmatics/linden-1", "speechmatics/oak-1", retired]


def speechmatics_language_hints() -> list[SpeechmaticsOptions]:
    return [
        SpeechmaticsOptions(language_hints=["en", "fr"]),
        SpeechmaticsOptions(language_hints="en"),  # type: ignore[typeddict-item]
    ]
