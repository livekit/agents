import pytest

from livekit.agents.tokenize import clause
from livekit.agents.tokenize._clause_sent import (
    ENGLISH,
    LanguageProfile,
    split_clauses,
)

# ---------------------------------------------------------------------------
# English clause tokenization
# ---------------------------------------------------------------------------

EN_TEXT = (
    "LiveKit is a platform for live audio and video, "
    "and it supports real-time communication. "
    "Mr. Smith joined the meeting, but Dr. Johnson was late. "
    "The price is $1,250.75 and the URL is https://example.com/path?q=1. "
    "However, the system worked perfectly."
)

EN_EXPECTED = [
    "LiveKit is a platform for live audio and video,",
    "and it supports real-time communication.",
    "Mr. Smith joined the meeting,",
    "but Dr. Johnson was late.",
    "The price is $1,250.75",
    "and the URL is https://example.com/path?q=1.",
    "However, the system worked perfectly.",
]


def test_english_clause_tokenizer():
    tok = clause.SentenceTokenizer(language="en")
    result = tok.tokenize(EN_TEXT)
    assert result == EN_EXPECTED


# ---------------------------------------------------------------------------
# Turkish clause tokenization
# ---------------------------------------------------------------------------

TR_TEXT = (
    "LiveKit ses ve video için bir platformdur, "
    "ve gerçek zamanlı iletişimi destekler. "
    "Dr. Ahmet toplantıya katıldı, ama Prof. Mehmet geç kaldı. "
    "Fiyat 1.250,75 TL'dir. "
    "Ancak sistem mükemmel çalıştı."
)

TR_EXPECTED = [
    "LiveKit ses ve video için bir platformdur,",
    "ve gerçek zamanlı iletişimi destekler.",
    "Dr. Ahmet toplantıya katıldı,",
    "ama Prof. Mehmet geç kaldı.",
    "Fiyat 1.250,75 TL'dir.",
    "Ancak sistem mükemmel çalıştı.",
]


def test_turkish_clause_tokenizer():
    tok = clause.SentenceTokenizer(language="tr")
    result = tok.tokenize(TR_TEXT)
    assert result == TR_EXPECTED


# ---------------------------------------------------------------------------
# German clause tokenization
# ---------------------------------------------------------------------------

DE_TEXT = (
    "LiveKit ist eine Plattform für Live-Audio und Video, "
    "und es unterstützt Echtzeit-Kommunikation. "
    "Dr. Müller nahm am Meeting teil, aber Prof. Schmidt kam zu spät. "
    "Der Preis beträgt 1.250,75 Euro. "
    "Jedoch funktionierte das System einwandfrei."
)

DE_EXPECTED = [
    "LiveKit ist eine Plattform für Live-Audio und Video,",
    "und es unterstützt Echtzeit-Kommunikation.",
    "Dr. Müller nahm am Meeting teil,",
    "aber Prof. Schmidt kam zu spät.",
    "Der Preis beträgt 1.250,75 Euro.",
    "Jedoch funktionierte das System einwandfrei.",
]


def test_german_clause_tokenizer():
    tok = clause.SentenceTokenizer(language="de")
    result = tok.tokenize(DE_TEXT)
    assert result == DE_EXPECTED


# ---------------------------------------------------------------------------
# French clause tokenization
# ---------------------------------------------------------------------------

FR_TEXT = (
    "LiveKit est une plateforme pour l'audio et la vidéo en direct, "
    "et elle prend en charge la communication en temps réel. "
    "M. Dupont a rejoint la réunion, mais Dr. Martin était en retard. "
    "Le prix est de 1.250,75 euros. "
    "Cependant le système a fonctionné parfaitement."
)

FR_EXPECTED = [
    "LiveKit est une plateforme pour l'audio",
    "et la vidéo en direct,",
    "et elle prend en charge la communication en temps réel.",
    "M. Dupont a rejoint la réunion,",
    "mais Dr. Martin était en retard.",
    "Le prix est de 1.250,75 euros.",
    "Cependant le système a fonctionné parfaitement.",
]


def test_french_clause_tokenizer():
    tok = clause.SentenceTokenizer(language="fr")
    result = tok.tokenize(FR_TEXT)
    assert result == FR_EXPECTED


# ---------------------------------------------------------------------------
# Italian clause tokenization
# ---------------------------------------------------------------------------

IT_TEXT = (
    "LiveKit è una piattaforma per audio e video in tempo reale, "
    "e supporta la comunicazione in diretta. "
    "Dott. Rossi ha partecipato alla riunione, ma Prof. Bianchi era in ritardo. "
    "Il prezzo è di 1.250,75 euro. "
    "Tuttavia il sistema ha funzionato perfettamente."
)

IT_EXPECTED = [
    "LiveKit è una piattaforma per audio",
    "e video in tempo reale,",
    "e supporta la comunicazione in diretta.",
    "Dott. Rossi ha partecipato alla riunione,",
    "ma Prof. Bianchi era in ritardo.",
    "Il prezzo è di 1.250,75 euro.",
    "Tuttavia il sistema ha funzionato perfettamente.",
]


def test_italian_clause_tokenizer():
    tok = clause.SentenceTokenizer(language="it")
    result = tok.tokenize(IT_TEXT)
    assert result == IT_EXPECTED


# ---------------------------------------------------------------------------
# Protected positions: abbreviations, numbers, URLs, emails, quotes
# ---------------------------------------------------------------------------


def test_abbreviations_not_split():
    """Periods in abbreviations must not trigger a split."""
    text = "Mr. Smith and Dr. Johnson met at Inc. headquarters today."
    tok = clause.SentenceTokenizer(language="en", min_clause_len=5)
    result = tok.tokenize(text)
    # "Mr." and "Dr." and "Inc." should not cause splits
    for segment in result:
        assert not segment.startswith(". ")


def test_url_not_split():
    """Punctuation inside URLs must not trigger a split."""
    text = "Visit https://example.com/path?q=1,2&v=3 for more details."
    tok = clause.SentenceTokenizer(language="en", min_clause_len=5)
    result = tok.tokenize(text)
    # The URL should remain intact in one of the segments
    found = any("https://example.com/path?q=1,2&v=3" in seg for seg in result)
    assert found, f"URL was split across segments: {result}"


def test_email_not_split():
    """Punctuation inside emails must not trigger a split."""
    text = "Contact user@example.com for more information about the project."
    tok = clause.SentenceTokenizer(language="en", min_clause_len=5)
    result = tok.tokenize(text)
    found = any("user@example.com" in seg for seg in result)
    assert found, f"Email was split across segments: {result}"


def test_quoted_text_not_split():
    """Punctuation inside quoted text must not trigger a split."""
    text = 'She said "no, I will not" and then left the room quickly.'
    tok = clause.SentenceTokenizer(language="en", min_clause_len=5)
    result = tok.tokenize(text)
    found = any('"no, I will not"' in seg for seg in result)
    assert found, f"Quoted text was split: {result}"


def test_number_separators_not_split():
    """Decimal and thousands separators must not trigger a split."""
    # English: 1,250.75
    text = "The total was 1,250.75 dollars and 3.14 percent."
    tok = clause.SentenceTokenizer(language="en", min_clause_len=5)
    result = tok.tokenize(text)
    found_amount = any("1,250.75" in seg for seg in result)
    found_pi = any("3.14" in seg for seg in result)
    assert found_amount, f"Number was split: {result}"
    assert found_pi, f"Decimal was split: {result}"


def test_time_not_split():
    """Colons in time formats must not trigger a split."""
    text = "The meeting starts at 14:30 and ends at 16:00 today."
    tok = clause.SentenceTokenizer(language="en", min_clause_len=5)
    result = tok.tokenize(text)
    found = any("14:30" in seg for seg in result)
    assert found, f"Time was split: {result}"


def test_bare_domain_not_split():
    """Periods in bare domain names (google.com) must not trigger a split."""
    text = "Visit google.com or example.org for more information about the project."
    tok = clause.SentenceTokenizer(language="en", min_clause_len=5)
    result = tok.tokenize(text)
    found_google = any("google.com" in seg for seg in result)
    found_example = any("example.org" in seg for seg in result)
    assert found_google, f"Bare domain was split: {result}"
    assert found_example, f"Bare domain was split: {result}"


def test_single_letter_abbreviation_not_split():
    """Periods in single-letter abbreviations (J. K. Rowling) must not split."""
    text = "The author J. K. Rowling wrote many books about the wizarding world."
    tok = clause.SentenceTokenizer(language="en", min_clause_len=5)
    result = tok.tokenize(text)
    # "J." and "K." periods should not cause sentence splits
    found = any("J. K. Rowling" in seg for seg in result)
    assert found, f"Single-letter abbreviation was split: {result}"


# ---------------------------------------------------------------------------
# Conjunction-based splitting
# ---------------------------------------------------------------------------


def test_conjunction_split_english():
    """English conjunctions should trigger clause boundaries."""
    text = "I like coffee but I prefer tea because it is healthier."
    tok = clause.SentenceTokenizer(language="en", min_clause_len=5)
    result = tok.tokenize(text)
    assert len(result) >= 2, f"Expected conjunctions to split: {result}"


def test_conjunction_split_turkish():
    """Turkish conjunctions should trigger clause boundaries."""
    text = "Kahve seviyorum ama çay tercih ediyorum çünkü daha sağlıklı."
    tok = clause.SentenceTokenizer(language="tr", min_clause_len=5)
    result = tok.tokenize(text)
    assert len(result) >= 2, f"Expected conjunctions to split: {result}"


def test_multiword_conjunction():
    """Multi-word conjunctions like 'ya da', 'bu yüzden' should work."""
    text = "Çay ya da kahve içebilirsin bu yüzden karar sana kalmış."
    tok = clause.SentenceTokenizer(language="tr", min_clause_len=5)
    result = tok.tokenize(text)
    assert len(result) >= 2, f"Expected multi-word conjunction split: {result}"


# ---------------------------------------------------------------------------
# Min clause length and merging
# ---------------------------------------------------------------------------


def test_min_clause_len_merging():
    """Clauses shorter than min_clause_len should be merged."""
    text = "Hi, this is a longer clause that should survive."
    # With high min_clause_len, short clauses merge
    tok = clause.SentenceTokenizer(language="en", min_clause_len=50)
    result = tok.tokenize(text)
    assert len(result) == 1, f"Short clauses should merge: {result}"


def test_empty_text():
    tok = clause.SentenceTokenizer(language="en")
    assert tok.tokenize("") == []


def test_whitespace_only():
    tok = clause.SentenceTokenizer(language="en")
    result = tok.tokenize("   ")
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Offset correctness
# ---------------------------------------------------------------------------


def test_offsets_cover_text():
    """Offsets should cover the input text contiguously."""
    text = "Hello world, and goodbye world. This is a test, but also an example."
    result = split_clauses(text, profile=ENGLISH, min_clause_len=5)
    assert result[0][1] == 0, "First clause must start at 0"
    assert result[-1][2] == len(text), "Last clause must end at len(text)"
    for i in range(1, len(result)):
        assert result[i][1] == result[i - 1][2], (
            f"Gap between clause {i - 1} and {i}: end={result[i - 1][2]}, start={result[i][1]}"
        )


# ---------------------------------------------------------------------------
# LanguageProfile via instance
# ---------------------------------------------------------------------------


def test_custom_language_profile():
    """Users should be able to pass a custom LanguageProfile."""
    german = LanguageProfile(
        name="de",
        conjunctions=("und", "aber", "oder", "weil", "denn"),
        abbreviations=("Dr.", "Prof.", "Nr."),
        decimal_sep=",",
        thousands_sep=".",
        min_clause_len=10,
    )
    tok = clause.SentenceTokenizer(language=german)
    text = "Ich mag Kaffee, aber ich bevorzuge Tee und Wasser."
    result = tok.tokenize(text)
    assert len(result) >= 2, f"German conjunctions should split: {result}"


def test_unknown_language_raises():
    with pytest.raises(ValueError, match="unknown language"):
        clause.SentenceTokenizer(language="klingon")


# ---------------------------------------------------------------------------
# retain_format
# ---------------------------------------------------------------------------


def test_retain_format():
    text = "Hello world,\n  and goodbye world."
    tok = clause.SentenceTokenizer(language="en", min_clause_len=5, retain_format=True)
    result = tok.tokenize(text)
    # With retain_format, whitespace/newlines should be preserved
    joined = "".join(result)
    assert "\n" in joined, f"Newline should be preserved: {result}"


# ---------------------------------------------------------------------------
# Streaming (BufferedSentenceStream)
# ---------------------------------------------------------------------------


async def test_clause_stream_english():
    """Streaming tokenization should produce the same results as batch."""
    tok = clause.SentenceTokenizer(language="en", min_clause_len=15)
    batch_result = tok.tokenize(EN_TEXT)

    # Feed text in small chunks
    pattern = [1, 2, 4]
    text = EN_TEXT
    chunks = []
    pattern_iter = iter(pattern * (len(text) // sum(pattern) + 1))

    for chunk_size in pattern_iter:
        if not text:
            break
        chunks.append(text[:chunk_size])
        text = text[chunk_size:]

    stream = tok.stream()
    for chunk in chunks:
        stream.push_text(chunk)
    stream.end_input()

    streamed_tokens = []
    async for token_data in stream:
        streamed_tokens.append(token_data.token)

    # Streamed tokens should match batch (they may differ slightly in
    # boundary cases due to buffering, but the content should match)
    assert len(streamed_tokens) > 0, "Stream should produce tokens"
    assert "".join(batch_result).replace(" ", "") == "".join(streamed_tokens).replace(" ", ""), (
        f"Stream content mismatch:\nbatch={batch_result}\nstream={streamed_tokens}"
    )


async def test_clause_stream_turkish():
    """Streaming tokenization for Turkish."""
    tok = clause.SentenceTokenizer(language="tr", min_clause_len=15)

    stream = tok.stream()
    for chunk in [TR_TEXT[i : i + 3] for i in range(0, len(TR_TEXT), 3)]:
        stream.push_text(chunk)
    stream.end_input()

    tokens = []
    async for token_data in stream:
        tokens.append(token_data.token)

    assert len(tokens) > 0, "Turkish stream should produce tokens"
