import re

from . import tokenizer


def split_words(
    text: str, ignore_punctuation: bool = True
) -> list[tuple[str, int, int]]:
    """
    Split the text into words.
    Returns a list of words with their start and end indices of the original text.
    """
    matches = re.finditer(r"\S+", text)
    words: list[tuple[str, int, int]] = []

    for match in matches:
        word = match.group(0)
        start_pos = match.start()
        end_pos = match.end()

        if ignore_punctuation:
            # TODO(theomonnom): acronyms passthrough
            translation_table = str.maketrans("", "", "".join(tokenizer.PUNCTUATIONS))
            word = word.translate(translation_table)

            if not word:
                continue

        words.append((word, start_pos, end_pos))

    return words
