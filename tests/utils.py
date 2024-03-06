from collections import defaultdict


def compare_word_counts(actual: str, expected: str):
    """Crude way to compare two strings by counting similar words."""

    lookup = defaultdict(int)

    for word in expected.split():
        lookup[word] += 1

    for word in actual.split():
        lookup[word] -= 1

    deviation = 0
    for word in lookup.keys():
        deviation += abs(lookup[word])

    # If every words is different, we want 0
    # if every word is the same, we want 1
    unique_words = len(lookup.keys())

    if unique_words == 0:
        return 0

    return 1 - (deviation / unique_words)
