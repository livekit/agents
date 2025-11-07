# InferEdge Moss plugin for LiveKit

This package wires the [InferEdge Moss](https://inferedge.dev/) semantic search
SDK into the LiveKit Agents plugin ecosystem. It exposes a thin wrapper around
``inferedge_moss`` with LiveKit-friendly defaults, environment-based
configuration, and index caching.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The plugin depends on the InferEdge client:

```bash
python -m pip install inferedge-moss
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
    print([doc.id for doc in result.docs])


asyncio.run(main())
```

See ``examples/dev/MossLifecycle.py`` for a full lifecycle demonstration.
