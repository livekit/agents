from typing import Literal

# https://docs.cohere.com/docs/models

ChatModels = Literal[
    "command-r-plus-08-2024",
    "command-r-08-2024",
    "command-r",
    "command",
    "command-nightly",
    "command-light",
    "command-light-nightly",
]

EmbeddingModels = Literal[
    "embed-english-v3.0",
    "embed-multilingual-v3.0",
    "embed-english-light-v3.0",
    "embed-multilingual-light-v3.0",
    "embed-english-v2.0",
    "embed-english-light-v2.0",
    "embed-multilingual-v2.0",
]
