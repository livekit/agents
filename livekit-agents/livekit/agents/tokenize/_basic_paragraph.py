import re


def split_paragraphs(text: str) -> list[tuple[str, int, int]]:
    """
    Split the text into paragraphs.
    Returns a list of paragraphs with their start and end indices of the original text.
    """
    # Use a regex pattern to split on one or more blank lines
    pattern = r"\n\s*\n"

    # Find all splits in the text
    splits = list(re.finditer(pattern, text))

    paragraphs = []
    start = 0

    # Handle the case where there are no splits (i.e., single paragraph)
    if not splits:
        return [(text.strip(), 0, len(text))]

    # Process each split
    for split in splits:
        end = split.start()
        paragraph = text[start:end].strip()
        if paragraph:  # Only add non-empty paragraphs
            paragraphs.append((paragraph, start, end))
        start = split.end()

    # Add the last paragraph
    last_paragraph = text[start:].strip()
    if last_paragraph:
        paragraphs.append((last_paragraph, start, len(text)))

    return paragraphs
