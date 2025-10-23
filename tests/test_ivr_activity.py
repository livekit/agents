from livekit.agents.voice.ivr.ivr_activity import TfidfLoopDetector


def _count_loops(transcripts: list[str], detector: TfidfLoopDetector | None = None) -> int:
    detector = detector or TfidfLoopDetector()
    loop_detected = 0
    for transcript in transcripts:
        detector.add_chunk(transcript)
        if detector.check_loop_detection():
            loop_detected += 1
    return loop_detected


def test_tfidf_no_loop_on_unique_user_speech() -> None:
    """Does not report loops when every utterance is unique."""

    transcripts = [
        "Welcome to automated phone system",
        "Type 1 for sales",
        "Type 2 for support",
        "Type 3 for billing",
        "Type 4 for technical support",
        "Type 5 for account management",
        "Type 6 for billing",
        "Type 7 for user management",
    ]

    assert _count_loops(transcripts) == 0


def test_tfidf_detects_loop_on_repeated_user_speech() -> None:
    """Reports a loop once similar prompts repeat enough times."""

    transcripts = [
        "Welcome to automated phone system",
        "Type 1 for sales",
        "Type 2 for support",
        "Type 3 for billing",
        "Type 4 for technical support",
        "Welcome to automated phone system",  # similar 1
        "Type 1 for sales",  # similar 2
        "Type 2 for support",  # similar 3, loop detected
        "Type 3 for billing",  # similar 4, loop detected
        "Type 4 for technical support",  # similar 5, loop detected
    ]

    assert _count_loops(transcripts) == 3


def test_tfidf_resets_after_novel_speech() -> None:
    """Resets its consecutive counter after a novel utterance and can detect again."""

    transcripts = [
        "Welcome to automated phone system",
        "Press 1 for sales",
        "Press 2 for support",
        "Press 1 for sales",  # similar 1
        "Press 2 for support",  # similar 2
        "Press 1 for sales",  # triggers first detection
        "Here's a new announcement never heard before",
        "Press 2 for support",  # similar 1 (after reset)
        "Press 1 for sales",  # similar 2
        "Press 2 for support",  # triggers second detection
    ]

    assert _count_loops(transcripts) == 2


def test_tfidf_handles_minor_phrase_variations() -> None:
    """Treats small textual variations as loops."""

    transcripts = [
        "Welcome to automated phone system",
        "Type 1 for sales, type 2 for support, type 3 for billing",
        "Again, type 1 for sales, type 2 for support, type 3 for billing",
        "And again, type 1 for sales, type 2 for support, type 3 for billing",
        "Repeat, type 1 for sales, type 2 for support, and type 3 for billing",
    ]

    assert _count_loops(transcripts) >= 1
