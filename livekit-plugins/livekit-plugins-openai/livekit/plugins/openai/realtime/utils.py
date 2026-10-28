"""OpenAI Realtime conversion helpers; defined in :mod:`livekit.agents.llm._realtime.openai_utils`."""

from livekit.agents.llm._realtime.openai_utils import (
    AZURE_DEFAULT_INPUT_AUDIO_TRANSCRIPTION as AZURE_DEFAULT_INPUT_AUDIO_TRANSCRIPTION,
    AZURE_DEFAULT_TURN_DETECTION as AZURE_DEFAULT_TURN_DETECTION,
    DEFAULT_INPUT_AUDIO_TRANSCRIPTION as DEFAULT_INPUT_AUDIO_TRANSCRIPTION,
    DEFAULT_MAX_RESPONSE_OUTPUT_TOKENS as DEFAULT_MAX_RESPONSE_OUTPUT_TOKENS,
    DEFAULT_MAX_SESSION_DURATION as DEFAULT_MAX_SESSION_DURATION,
    DEFAULT_TOOL_CHOICE as DEFAULT_TOOL_CHOICE,
    DEFAULT_TURN_DETECTION as DEFAULT_TURN_DETECTION,
    calculate_confidence_from_logprobs as calculate_confidence_from_logprobs,
    livekit_item_to_openai_item as livekit_item_to_openai_item,
    openai_item_to_livekit_item as openai_item_to_livekit_item,
    to_audio_transcription as to_audio_transcription,
    to_noise_reduction as to_noise_reduction,
    to_oai_tool_choice as to_oai_tool_choice,
    to_turn_detection as to_turn_detection,
)
