def split_paragraphs(text: str) -> list[str]:
    sep = "\n\n"

    paragraphs = text.split(sep)
    new_paragraphs = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        new_paragraphs.append(p)

    return new_paragraphs
