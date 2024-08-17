import re


def split_words(text: str, ignore_punctuation: bool = True) -> list[str]:
    # fmt: off
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>',
                    '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '±', '—', '‘', '’', '“', '”', '…']

    # fmt: on

    if ignore_punctuation:
        # TODO(theomonnom): Ignore acronyms
        translation_table = str.maketrans("", "", "".join(punctuations))
        text = text.translate(translation_table)

    words = re.split(r"\s+", text)
    new_words = []
    for word in words:
        if not word:
            continue  # ignore empty
        new_words.append(word)

    return new_words
