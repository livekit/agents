# tests/test_handler_unit.py
from extensions.interrupt_handler.handler import is_filler_only, tokenize


def test_tokenize_and_basic():
    assert tokenize("Uh, umm!") == ["uh", "umm"]


def test_is_filler_true_high_conf():
    assert is_filler_only("uh umm", confidence=0.8) is True


def test_is_filler_false_command():
    assert is_filler_only("uh stop", confidence=0.9) is False


def test_is_filler_false_nonfiller_word():
    assert is_filler_only("uh yes", confidence=0.9) is False


def test_empty_transcript():
    assert is_filler_only("", confidence=0.5) is True
