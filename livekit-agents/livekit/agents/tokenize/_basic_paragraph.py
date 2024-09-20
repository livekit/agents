import re


def split_paragraphs(text: str) -> list[tuple[str, int, int]]:
    """
    Split the text into paragraphs.
    Returns a list of paragraphs with their start and end indices of the original text.
    """
    matches = re.finditer(r"\n{2,}", text)
    paragraphs = []

    for match in matches:
        paragraph = match.group(0)
        start_pos = match.start()
        end_pos = match.end()
        paragraphs.append((paragraph.strip(), start_pos, end_pos))

    return paragraphs
