import os


def get_base_url(base_url: str | None) -> str:
    if base_url is None:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return base_url
