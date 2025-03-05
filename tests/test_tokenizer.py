import pytest
from livekit.agents import tokenize
from livekit.agents.tokenize import basic
from livekit.agents.tokenize._basic_paragraph import split_paragraphs
from livekit.plugins import nltk

# Download the punkt tokenizer, will only download if not already present
nltk.NltkPlugin().download_files()

TEXT = (
    "Hi! "
    "LiveKit is a platform for live audio and video applications and services. \n\n"
    "R.T.C stands for Real-Time Communication... again R.T.C. "
    "Mr. Theo is testing the sentence tokenizer. "
    "\nThis is a test. Another test. "
    "A short sentence.\n"
    "A longer sentence that is longer than the previous sentence. "
    "f(x) = x * 2.54 + 42. "
    "Hey!\n Hi! Hello! "
    "\n\n"
)

EXPECTED_MIN_20 = [
    "Hi! LiveKit is a platform for live audio and video applications and services.",
    "R.T.C stands for Real-Time Communication... again R.T.C.",
    "Mr. Theo is testing the sentence tokenizer.",
    "This is a test. Another test.",
    "A short sentence. A longer sentence that is longer than the previous sentence.",
    "f(x) = x * 2.54 + 42.",
    "Hey! Hi! Hello!",
]

EXPECTED_MIN_20_RETAIN_FORMAT = [
    "Hi! LiveKit is a platform for live audio and video applications and services.",
    " \n\nR.T.C stands for Real-Time Communication... again R.T.C.",
    " Mr. Theo is testing the sentence tokenizer.",
    " \nThis is a test. Another test.",
    " A short sentence.\nA longer sentence that is longer than the previous sentence.",
    " f(x) = x * 2.54 + 42.",
    " Hey!\n Hi! Hello! \n\n",
]

SENT_TOKENIZERS = [
    (nltk.SentenceTokenizer(min_sentence_len=20), EXPECTED_MIN_20),
    (basic.SentenceTokenizer(min_sentence_len=20), EXPECTED_MIN_20),
    (
        basic.SentenceTokenizer(min_sentence_len=20, retain_format=True),
        EXPECTED_MIN_20_RETAIN_FORMAT,
    ),
]


@pytest.mark.parametrize("tokenizer, expected", SENT_TOKENIZERS)
def test_sent_tokenizer(tokenizer: tokenize.SentenceTokenizer, expected: list[str]):
    segmented = tokenizer.tokenize(text=TEXT)
    for i, segment in enumerate(expected):
        assert segment == segmented[i]


@pytest.mark.parametrize("tokenizer, expected", SENT_TOKENIZERS)
async def test_streamed_sent_tokenizer(tokenizer: tokenize.SentenceTokenizer, expected: list[str]):
    # divide text by chunks of arbitrary length (1-4)
    pattern = [1, 2, 4]
    text = TEXT
    chunks = []
    pattern_iter = iter(pattern * (len(text) // sum(pattern) + 1))

    for chunk_size in pattern_iter:
        if not text:
            break
        chunks.append(text[:chunk_size])
        text = text[chunk_size:]

    stream = tokenizer.stream()
    for chunk in chunks:
        stream.push_text(chunk)

    stream.end_input()

    for i in range(len(expected)):
        ev = await stream.__anext__()
        assert ev.token == expected[i]


WORDS_TEXT = "This is a test. Blabla another test! multiple consecutive spaces:     done"
WORDS_EXPECTED = [
    "This",
    "is",
    "a",
    "test",
    "Blabla",
    "another",
    "test",
    "multiple",
    "consecutive",
    "spaces",
    "done",
]

WORD_TOKENIZERS = [basic.WordTokenizer()]


@pytest.mark.parametrize("tokenizer", WORD_TOKENIZERS)
def test_word_tokenizer(tokenizer: tokenize.WordTokenizer):
    tokens = tokenizer.tokenize(text=WORDS_TEXT)
    for i, token in enumerate(WORDS_EXPECTED):
        assert token == tokens[i]


@pytest.mark.parametrize("tokenizer", WORD_TOKENIZERS)
async def test_streamed_word_tokenizer(tokenizer: tokenize.WordTokenizer):
    # divide text by chunks of arbitrary length (1-4)
    pattern = [1, 2, 4]
    text = WORDS_TEXT
    chunks = []
    pattern_iter = iter(pattern * (len(text) // sum(pattern) + 1))

    for chunk_size in pattern_iter:
        if not text:
            break
        chunks.append(text[:chunk_size])
        text = text[chunk_size:]

    stream = tokenizer.stream()
    for chunk in chunks:
        stream.push_text(chunk)

    stream.end_input()

    for i in range(len(WORDS_EXPECTED)):
        ev = await stream.__anext__()
        assert ev.token == WORDS_EXPECTED[i]


WORDS_PUNCT_TEXT = 'This is <phoneme alphabet="cmu-arpabet" ph="AE K CH UW AH L IY">actually</phoneme> tricky to handle.'

WORDS_PUNCT_EXPECTED = [
    "This",
    "is",
    "<phoneme",
    'alphabet="cmu-arpabet"',
    'ph="AE',
    "K",
    "CH",
    "UW",
    "AH",
    "L",
    'IY">actually</phoneme>',
    "tricky",
    "to",
    "handle.",
]

WORD_PUNCT_TOKENIZERS = [basic.WordTokenizer(ignore_punctuation=False)]


@pytest.mark.parametrize("tokenizer", WORD_PUNCT_TOKENIZERS)
def test_punct_word_tokenizer(tokenizer: tokenize.WordTokenizer):
    tokens = tokenizer.tokenize(text=WORDS_PUNCT_TEXT)
    for i, token in enumerate(WORDS_PUNCT_EXPECTED):
        assert token == tokens[i]


@pytest.mark.parametrize("tokenizer", WORD_PUNCT_TOKENIZERS)
async def test_streamed_punct_word_tokenizer(tokenizer: tokenize.WordTokenizer):
    # divide text by chunks of arbitrary length (1-4)
    pattern = [1, 2, 4]
    text = WORDS_PUNCT_TEXT
    chunks = []
    pattern_iter = iter(pattern * (len(text) // sum(pattern) + 1))

    for chunk_size in pattern_iter:
        if not text:
            break
        chunks.append(text[:chunk_size])
        text = text[chunk_size:]

    stream = tokenizer.stream()
    for chunk in chunks:
        stream.push_text(chunk)

    stream.end_input()

    for i in range(len(WORDS_PUNCT_EXPECTED)):
        ev = await stream.__anext__()
        assert ev.token == WORDS_PUNCT_EXPECTED[i]


HYPHENATOR_TEXT = [
    "Segment",
    "expected",
    "communication",
    "window",
    "welcome",
    "bedroom",
]

HYPHENATOR_EXPECTED = [
    ["Seg", "ment"],
    ["ex", "pect", "ed"],
    ["com", "mu", "ni", "ca", "tion"],
    ["win", "dow"],
    ["wel", "come"],
    ["bed", "room"],
]


def test_hyphenate_word():
    for i, word in enumerate(HYPHENATOR_TEXT):
        hyphenated = basic.hyphenate_word(word)
        assert hyphenated == HYPHENATOR_EXPECTED[i]


REPLACE_TEXT = (
    "This is a test. Hello world, I'm creating this agents..     framework. Once again "
    "framework.  A.B.C"
)
REPLACE_EXPECTED = (
    "This is a test. Hello universe, I'm creating this assistants..     library. twice again "
    "library.  A.B.C.D"
)

REPLACE_REPLACEMENTS = {
    "world": "universe",
    "framework": "library",
    "a.b.c": "A.B.C.D",
    "once": "twice",
    "agents": "assistants",
}


def test_replace_words():
    replaced = tokenize.utils.replace_words(text=REPLACE_TEXT, replacements=REPLACE_REPLACEMENTS)
    assert replaced == REPLACE_EXPECTED


async def test_replace_words_async():
    pattern = [1, 2, 4]
    text = REPLACE_TEXT
    chunks = []
    pattern_iter = iter(pattern * (len(text) // sum(pattern) + 1))

    for chunk_size in pattern_iter:
        if not text:
            break
        chunks.append(text[:chunk_size])
        text = text[chunk_size:]

    async def _replace_words_async():
        for chunk in chunks:
            yield chunk

    replaced_chunks = []

    async for chunk in tokenize.utils.replace_words(
        text=_replace_words_async(), replacements=REPLACE_REPLACEMENTS
    ):
        replaced_chunks.append(chunk)

    replaced = "".join(replaced_chunks)
    assert replaced == REPLACE_EXPECTED


PARAGRAPH_TEST_CASES = [
    ("Single paragraph.", [("Single paragraph.", 0, 17)]),
    (
        "Paragraph 1.\n\nParagraph 2.",
        [("Paragraph 1.", 0, 12), ("Paragraph 2.", 14, 26)],
    ),
    (
        "Para 1.\n\nPara 2.\n\nPara 3.",
        [("Para 1.", 0, 7), ("Para 2.", 9, 16), ("Para 3.", 18, 25)],
    ),
    (
        "\n\nParagraph with leading newlines.",
        [("Paragraph with leading newlines.", 2, 34)],
    ),
    (
        "Paragraph with trailing newlines.\n\n",
        [("Paragraph with trailing newlines.", 0, 33)],
    ),
    (
        "\n\n  Paragraph with leading and trailing spaces.  \n\n",
        [("Paragraph with leading and trailing spaces.", 4, 47)],
    ),
    (
        "Para 1.\n\n\n\nPara 2.",  # Multiple newlines between paragraphs
        [("Para 1.", 0, 7), ("Para 2.", 11, 18)],
    ),
    (
        "Para 1.\n \n \nPara 2.",  # Newlines with spaces between paragraphs
        [("Para 1.", 0, 7), ("Para 2.", 12, 19)],
    ),
    (
        "",  # Empty string
        [],
    ),
    (
        "\n\n\n",  # Only newlines
        [],
    ),
    (
        "Line 1\nLine 2\nLine 3",  # Single paragraph with newlines
        [("Line 1\nLine 2\nLine 3", 0, 20)],
    ),
]


@pytest.mark.parametrize(
    "test_case",
    PARAGRAPH_TEST_CASES,
)
def test_split_paragraphs(test_case):
    input_text, expected_output = test_case
    result = split_paragraphs(input_text)
    assert result == expected_output, f"Failed for input: {input_text}"
