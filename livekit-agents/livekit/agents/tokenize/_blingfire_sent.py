from .blingfire import text_to_sentences_and_offsets


def split_sentences(
    text: str,
    min_sentence_len: int = 20,
) -> list[tuple[str, int, int]]:
    _, offsets = text_to_sentences_and_offsets(text)
    results: list[tuple[str, int, int]] = []
    # blingfire skips the spaces between sentences, so we need to keep track of the last end
    last_end = 0
    for _, end in offsets:
        if not results or len(results[-1][0]) >= min_sentence_len:
            results.append((text[last_end:end], last_end, end))
            last_end = end
            continue
        _, last_start, _ = results[-1]
        results[-1] = (
            text[last_start:end],
            last_start,
            end,
        )
        last_end = end

    if 0 < last_end < len(text):
        results[-1] = (text[results[-1][1] :], results[-1][1], len(text))

    return results


if __name__ == "__main__":
    from .blingfire import text_to_sentences

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
    print(split_sentences(TEXT, min_sentence_len=1))
    print(text_to_sentences(TEXT))
