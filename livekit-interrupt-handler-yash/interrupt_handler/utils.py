import re

def normalize_text(text: str):
    """
    Lowercases, strips punctuation, splits into tokens.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = text.split()
    return tokens
