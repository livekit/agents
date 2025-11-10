import pytest
from ..src.config import IHConfig
from ..src.classifier import UtteranceClassifier

@pytest.fixture
def clf():
    return UtteranceClassifier(IHConfig())

def test_filler_only(clf):
    j = clf.decide("umm hmm", 0.95, 500)
    assert j.label == "FILLER_ONLY"

def test_low_confidence(clf):
    j = clf.decide("hello", 0.1, 500)
    assert j.label == "LOW_CONF"

def test_hard_intent(clf):
    j = clf.decide("stop please", 0.9, 500)
    assert j.label == "HARD_INTENT"

def test_contentful(clf):
    j = clf.decide("this looks good", 0.9, 500)
    assert j.label == "CONTENT"
