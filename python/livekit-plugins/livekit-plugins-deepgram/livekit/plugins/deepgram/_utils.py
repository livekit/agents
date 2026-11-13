from collections.abc import Mapping, Sequence
from urllib.parse import urlencode


def to_deepgram_url(
    opts: Mapping[str, object],
    base_url: str,
    *,
    websocket: bool,
    keywords: Sequence[tuple[str, float]] | None = None,
    replace: Mapping[str, str] | None = None,
) -> str:
    query = {
        key: str(value).lower() if isinstance(value, bool) else value for key, value in opts.items()
    }
    if keywords:
        query["keywords"] = [f"{keyword}:{intensifier}" for keyword, intensifier in keywords]
    if replace:
        query["replace"] = [f"{term}:{replacement}" for term, replacement in replace.items()]

    if websocket and base_url.startswith("http"):
        base_url = base_url.replace("http", "ws", 1)

    elif not websocket and base_url.startswith("ws"):
        base_url = base_url.replace("ws", "http", 1)
    return f"{base_url}?{urlencode(query, doseq=True)}"
