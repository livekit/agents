from livekit.plugins import nltk

TEXT = (
    "Hi! "
    "LiveKit is a platform for live audio and video applications and services. "
    "R.T.C stands for Real-Time Communication... again R.T.C. "
    "Mr. Theo is testing the sentence tokenizer. "
    "This is a test. Another test. "
    "A short sentence. "
    "A longer sentence that is longer than the previous sentence. "
    "f(x) = x * 2.54 + 42. "
    "Hey! Hi! Hello! "
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


def test_sent_tokenizer():
    sentence_tokenizer = nltk.SentenceTokenizer()
    segmented = sentence_tokenizer.tokenize(
        text=TEXT, language="english", min_sentence_len=20
    )
    for i, segment in enumerate(EXPECTED_MIN_20):
        assert segment == segmented[i].text


async def test_streamed_sent_tokenizer():
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

    sentence_tokenizer = nltk.SentenceTokenizer()
    stream = sentence_tokenizer.stream(language="english", min_sentence_len=20)
    for chunk in chunks:
        stream.push_text(chunk)

    for i in range(len(EXPECTED_MIN_20) - 1):
        segmented = await stream.__anext__()
        assert segmented.text == EXPECTED_MIN_20[i]

    await stream.flush()

    segmented = await stream.__anext__()
    assert segmented.text == EXPECTED_MIN_20[-1]
