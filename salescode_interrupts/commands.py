# salescode_interrupts/commands.py
# ---------------------------------------------------------------------------
# Factory for creating InterruptFilter instances with dynamic keyword lists
# ---------------------------------------------------------------------------

from salescode_interrupts.interrupt_filter import (
    InterruptFilter,
    IGNORED_FILLERS,
    INTERRUPT_COMMANDS,
    ASRSegment,
)


def build_interrupt_filter(session):
    """
    Creates a fully configured InterruptFilter for a LiveKit AgentSession.
    All filler words + interruption commands come from exported shared lists.
    """

    return InterruptFilter(
        is_agent_speaking=lambda: getattr(session, "is_tts_active", False),
        stop_agent_speaking=lambda: session.stop_audio_output(),

        # ✅ Callback behavior (you can swap with logging or metrics)
        on_valid_interrupt=lambda seg, dec: print(f"[INTERRUPT] {dec.cleaned_text}"),
        on_ignored_filler=lambda seg, dec: print(f"[FILLER IGNORED] {seg.cleaned_text}"),
        on_speech_when_quiet=lambda seg: print(f"[USER SPEECH] {seg.text}"),

        # ✅ Dynamic keyword lists
        ignored_words=IGNORED_FILLERS,
        interrupt_commands=INTERRUPT_COMMANDS,

        min_confidence=0.60,
    )
