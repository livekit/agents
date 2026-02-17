# Moss plugin for LiveKit Agents

This package wires the [Moss](https://www.moss.dev/) semantic search
SDK into the LiveKit Agents plugin ecosystem. It exposes a thin wrapper around
`moss` with LiveKit-friendly defaults, environment-based
configuration, and index caching.

## Installation

```bash
cd livekit-plugins/livekit-plugins-moss
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools build
python -m pip install -e .

# (Run the example shown below — moved to the end of this file for clarity.)
```

Set your credentials via environment variables:

```bash
export MOSS_PROJECT_ID="your-project-id"
export MOSS_PROJECT_KEY="your-project-key"
```

## Quick start

```python
import asyncio
from livekit.plugins.moss import DocumentInfo, MossClient


async def main() -> None:
    client = MossClient()
    await client.create_index(
        "demo",
        [DocumentInfo(id="doc1", text="Semantic search is fast")],
        model_id="moss-minilm",
    )
    await client.load_index("demo")
    result = await client.query("demo", "fast search")
    print([doc.text for doc in result.docs])


asyncio.run(main())
```
