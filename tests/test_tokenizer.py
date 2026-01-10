import pytest

from livekit.agents import tokenize
from livekit.agents.tokenize import basic, blingfire
from livekit.agents.tokenize._basic_paragraph import split_paragraphs
from livekit.agents.voice.transcription import filters
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
    "This is a sentence. 这是一个中文句子。これは日本語の文章です。"
    "你好！LiveKit是一个直播音频和视频应用程序和服务的平台。"
    "\nThis is a sentence contains   consecutive spaces."
)

EXPECTED_MIN_20 = [
    "Hi! LiveKit is a platform for live audio and video applications and services.",
    "R.T.C stands for Real-Time Communication... again R.T.C.",
    "Mr. Theo is testing the sentence tokenizer.",
    "This is a test. Another test.",
    "A short sentence. A longer sentence that is longer than the previous sentence.",
    "f(x) = x * 2.54 + 42.",
    "Hey! Hi! Hello! This is a sentence.",
    "这是一个中文句子。 これは日本語の文章です。",
    "你好！ LiveKit是一个直播音频和视频应用程序和服务的平台。",
    "This is a sentence contains   consecutive spaces.",
]

EXPECTED_MIN_20_RETAIN_FORMAT = [
    "Hi! LiveKit is a platform for live audio and video applications and services.",
    " \n\nR.T.C stands for Real-Time Communication... again R.T.C.",
    " Mr. Theo is testing the sentence tokenizer.",
    " \nThis is a test. Another test.",
    " A short sentence.\nA longer sentence that is longer than the previous sentence.",
    " f(x) = x * 2.54 + 42.",
    " Hey!\n Hi! Hello! \n\nThis is a sentence.",
    " 这是一个中文句子。これは日本語の文章です。",
    "你好！LiveKit是一个直播音频和视频应用程序和服务的平台。",
    "\nThis is a sentence contains   consecutive spaces.",
]

EXPECTED_MIN_20_NLTK = [
    "Hi! LiveKit is a platform for live audio and video applications and services.",
    "R.T.C stands for Real-Time Communication... again R.T.C.",
    "Mr. Theo is testing the sentence tokenizer.",
    "This is a test. Another test.",
    "A short sentence. A longer sentence that is longer than the previous sentence.",
    "f(x) = x * 2.54 + 42.",
    "Hey! Hi! Hello! This is a sentence.",
    # nltk does not support character-based languages like CJK
    "这是一个中文句子。これは日本語の文章です。你好！LiveKit是一个直播音频和视频应用程序和服务的平台。\nThis is a sentence contains   consecutive spaces.",  # noqa: E501
]

EXPECTED_MIN_20_BLINGFIRE = [
    "Hi! LiveKit is a platform for live audio and video applications and services.",
    "R.T.C stands for Real-Time Communication... again R.T.C. Mr. Theo is testing the sentence tokenizer.",
    "This is a test. Another test.",
    "A short sentence. A longer sentence that is longer than the previous sentence. f(x) = x * 2.54 + 42.",
    "Hey! Hi! Hello! This is a sentence.",
    "这是一个中文句子。これは日本語の文章です。",
    "你好！LiveKit是一个直播音频和视频应用程序和服务的平台。",
    "This is a sentence contains   consecutive spaces.",
]


SENT_TOKENIZERS = [
    (nltk.SentenceTokenizer(min_sentence_len=20), EXPECTED_MIN_20_NLTK),
    (basic.SentenceTokenizer(min_sentence_len=20), EXPECTED_MIN_20),
    (
        basic.SentenceTokenizer(min_sentence_len=20, retain_format=True),
        EXPECTED_MIN_20_RETAIN_FORMAT,
    ),
    (blingfire.SentenceTokenizer(min_sentence_len=20), EXPECTED_MIN_20_BLINGFIRE),
]


@pytest.mark.parametrize("tokenizer, expected", SENT_TOKENIZERS)
def test_sent_tokenizer(tokenizer: tokenize.SentenceTokenizer, expected: list[str]):
    segmented = tokenizer.tokenize(text=TEXT)
    print(segmented)
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


WORDS_PUNCT_TEXT = (
    'This is <phoneme alphabet="cmu-arpabet" ph="AE K CH UW AH L IY">actually</phoneme> tricky to handle.'  # noqa: E501
    "这是一个中文句子。 これは日本語の文章です。"
)

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
    "这",
    "是",
    "一",
    "个",
    "中",
    "文",
    "句",
    "子",
    "。",
    "こ",
    "れ",
    "は",
    "日",
    "本",
    "語",
    "の",
    "文",
    "章",
    "で",
    "す",
    "。",
]

WORD_PUNCT_TOKENIZERS = [basic.WordTokenizer(ignore_punctuation=False, split_character=True)]


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


# ============================================================================
# TTS Preprocessing Filter Tests
# ============================================================================


async def apply_filters(text: str, filter_names: list[str]) -> str:
    """Helper to apply filters to text."""

    async def text_stream():
        yield text

    filtered = filters.apply_text_transforms(text_stream(), filter_names)
    result = ""
    async for chunk in filtered:
        result += chunk
    return result


class TestNumberFormatting:
    """Test number formatting for TTS."""

    @pytest.mark.asyncio
    async def test_small_numbers_to_words(self):
        result = await apply_filters("I have 5 apples", ["format_numbers"])
        assert "five apples" in result

        result = await apply_filters("There are 23 oranges", ["format_numbers"])
        assert "twenty three oranges" in result

    @pytest.mark.asyncio
    async def test_large_numbers_remain_numeric(self):
        result = await apply_filters("Population is 1,234,567", ["format_numbers"])
        assert "1234567" in result
        assert "," not in result

    @pytest.mark.asyncio
    async def test_decimal_numbers(self):
        result = await apply_filters("The value is 3.14", ["format_numbers"])
        assert "3 point 1 4" in result

    @pytest.mark.asyncio
    async def test_negative_numbers(self):
        result = await apply_filters("Temperature is -5 degrees", ["format_numbers"])
        assert "minus five" in result

    @pytest.mark.asyncio
    async def test_years_unchanged(self):
        result = await apply_filters("In 2024 we saw progress", ["format_numbers"])
        assert "2024" in result


class TestCurrencyFormatting:
    """Test currency formatting for TTS."""

    @pytest.mark.asyncio
    async def test_whole_dollars(self):
        result = await apply_filters("It costs $42", ["format_dollar_amounts"])
        assert "forty two dollars" in result
        assert "$" not in result

    @pytest.mark.asyncio
    async def test_dollars_with_cents(self):
        result = await apply_filters("Price is $12.50", ["format_dollar_amounts"])
        assert "twelve dollars" in result
        assert "fifty cents" in result

    @pytest.mark.asyncio
    async def test_single_dollar(self):
        result = await apply_filters("Only $1", ["format_dollar_amounts"])
        assert "one dollar" in result
        # Should not have plural
        assert result.count("dollar") == 1


class TestPercentagesAndMeasurements:
    """Test percentages and measurement units."""

    @pytest.mark.asyncio
    async def test_percentages(self):
        result = await apply_filters("Discount is 15%", ["format_percentages"])
        assert "15 percent" in result
        assert "%" not in result

    @pytest.mark.asyncio
    async def test_distances(self):
        result = await apply_filters("It's 5 km away", ["format_distances"])
        assert "5 kilometers" in result
        assert "km" not in result

    @pytest.mark.asyncio
    async def test_weight_units(self):
        result = await apply_filters("Weighs 10 kg", ["format_units"])
        assert "ten kilograms" in result

    @pytest.mark.asyncio
    async def test_volume_units(self):
        result = await apply_filters("Add 2 l of water", ["format_units"])
        assert "two liters" in result


class TestCommunicationFormats:
    """Test phone numbers and email addresses."""

    @pytest.mark.asyncio
    async def test_phone_numbers(self):
        result = await apply_filters("Call 555-123-4567", ["format_phone_numbers"])
        assert "5 5 5" in result
        assert "-" not in result

    @pytest.mark.asyncio
    async def test_email_addresses(self):
        result = await apply_filters("Email john.doe@example.com", ["format_emails"])
        assert "at" in result
        assert "dot" in result
        assert "@" not in result


class TestDateTimeFormats:
    """Test date and time formatting."""

    @pytest.mark.asyncio
    async def test_date_formatting(self):
        result = await apply_filters("Meeting on 2024-12-25", ["format_dates"])
        assert "December" in result
        assert "2024" in result

    @pytest.mark.asyncio
    async def test_time_formatting(self):
        result = await apply_filters("Meet at 14:30", ["format_times"])
        # Should preserve time format (with COLON placeholder)
        assert "14" in result
        assert "30" in result


class TestAcronymsAndAbbreviations:
    """Test acronym handling."""

    @pytest.mark.asyncio
    async def test_known_acronyms_lowercase(self):
        result = await apply_filters("NASA and FBI use API", ["format_acronyms"])
        assert "nasa" in result
        assert "fbi" in result
        assert "api" in result

    @pytest.mark.asyncio
    async def test_acronyms_with_vowels(self):
        result = await apply_filters("Working with HTML and CSS", ["format_acronyms"])
        assert "html" in result
        assert "css" in result

    @pytest.mark.asyncio
    async def test_acronyms_without_vowels_spaced(self):
        result = await apply_filters("The XYZ protocol", ["format_acronyms"])
        # Should be spaced like "X Y Z"
        assert "XYZ" not in result


class TestNewlinesAndWhitespace:
    """Test newline and whitespace handling."""

    @pytest.mark.asyncio
    async def test_single_newline_becomes_space(self):
        result = await apply_filters("Line one\nLine two", ["replace_newlines_with_periods"])
        assert "\n" not in result
        assert "Line one Line two" in result

    @pytest.mark.asyncio
    async def test_multiple_newlines_single_period(self):
        result = await apply_filters("Para one\n\nPara two", ["replace_newlines_with_periods"])
        assert "Para one. Para two" in result


class TestCriticalDecimalBugs:
    """Test critical bugs with decimal number handling."""

    @pytest.mark.asyncio
    async def test_decimal_percentages_not_mixed(self):
        """Decimal percentages should not get mixed format like 'eighty nine.5'"""
        result = await apply_filters("Rate is 89.5%", ["format_percentages"])
        assert "eighty nine.5" not in result
        assert "89.5 percent" in result

    @pytest.mark.asyncio
    async def test_small_dollar_decimals(self):
        """Small dollar amounts like $0.023 should speak out each decimal digit"""
        result = await apply_filters("Price is $0.023", ["format_dollar_amounts"])
        assert "cents3" not in result
        assert "zero point zero two three dollars" in result

    @pytest.mark.asyncio
    async def test_decimal_weights_not_mixed(self):
        """Weight decimals should not get mixed format like 'eighty five.six'"""
        result = await apply_filters("Weight: 85.6 kg", ["format_units"])
        assert "eighty five.six" not in result
        assert "85.6 kilograms" in result

    @pytest.mark.asyncio
    async def test_decimal_distances_not_mixed(self):
        """Distance decimals should not get mixed format like 'fifteen.3'"""
        result = await apply_filters("Distance: 15.3 mi", ["format_distances"])
        assert "fifteen.3" not in result
        assert "15.3 miles" in result

    @pytest.mark.asyncio
    async def test_comma_separated_numbers_with_units(self):
        """Numbers with commas in measurements should not get broken"""
        result = await apply_filters("Altitude: 1,500 m", ["format_distances"])
        assert "one,500" not in result
        assert "1500 meters" in result


class TestStreamingFilters:
    """Test async streaming with chunked input."""

    @pytest.mark.asyncio
    async def test_streaming_numbers(self):
        """Test number formatting with streaming input."""

        async def text_stream():
            yield "I have "
            yield "5"
            yield " apples and "
            yield "23"
            yield " oranges"

        filtered = filters.apply_text_transforms(text_stream(), ["format_numbers"])
        result = ""
        async for chunk in filtered:
            result += chunk

        assert "five apples" in result
        assert "twenty three oranges" in result

    @pytest.mark.asyncio
    async def test_streaming_currency(self):
        """Test currency formatting with streaming input."""

        async def text_stream():
            yield "It costs $"
            yield "42"
            yield " today"

        filtered = filters.apply_text_transforms(text_stream(), ["format_dollar_amounts"])
        result = ""
        async for chunk in filtered:
            result += chunk

        assert "forty two dollars" in result

    @pytest.mark.asyncio
    async def test_streaming_email(self):
        """Test email formatting with streaming input."""

        async def text_stream():
            yield "Email john"
            yield ".doe@"
            yield "example.com"

        filtered = filters.apply_text_transforms(text_stream(), ["format_emails"])
        result = ""
        async for chunk in filtered:
            result += chunk

        assert "at" in result
        assert "dot" in result


class TestMultipleFilters:
    """Test applying multiple filters together."""

    @pytest.mark.asyncio
    async def test_combined_filters(self):
        """Test multiple filters applied in sequence."""
        text = "Meeting on 2024-12-25 with 15% discount. Call 555-123-4567."
        result = await apply_filters(
            text, ["format_dates", "format_percentages", "format_phone_numbers"]
        )

        assert "December" in result
        assert "percent" in result
        assert "5 5 5" in result
