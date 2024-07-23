import re


def split_words(text: str, ignore_punctuation: bool = True) -> list[str]:
    # fmt: off
    punctuations = [".", ",", "!", "?", ";", ":", "'", '"', "(", ")", "[", "]", "{", "}", "<", ">",
                    "—"]
    # fmt: on

    if ignore_punctuation:
        for p in punctuations:
            # TODO(theomonnom): Ignore acronyms
            text = text.replace(p, "")

    words = re.split("[ \n]+", text)
    new_words = []
    for word in words:
        if not word:
            continue  # ignore empty
        new_words.append(word)

    return new_words
