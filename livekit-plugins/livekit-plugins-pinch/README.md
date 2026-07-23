# livekit-plugins-pinch

Real-time voice translation for [LiveKit Agents](https://github.com/livekit/agents), powered by [Pinch](https://www.startpinch.com).

---

## What it does

Drop this plugin into any LiveKit room and it will:

- Translate spoken audio in real time from one language to another
- Publish the translated audio back into the room as a separate track
- Emit transcripts (original + translated) via a simple callback

---

## Installation

```bash
pip install livekit-plugins-pinch
```

---

## Compatibility

- Python >= 3.10
- livekit >= 0.12.0
- livekit-agents >= 0.8.0

---

## Configuration

You need a **Pinch API key**. Get one at the [developers portal](https://portal.startpinch.com/dashboard/developers).

Set it in your environment:

```bash
export PINCH_API_KEY=pk_your_key_here
```

That's the only credential this plugin needs. Your LiveKit credentials stay in your own app as usual.

---

## Usage

```python
from livekit import rtc
from livekit.plugins.pinch import Translator, TranslatorOptions

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    translator = Translator(
        options=TranslatorOptions(
            source_language="en-US",
            target_language="es-ES",
            voice_type="clone",  # "clone" | "female" | "male"
        )
    )

    @translator.on_transcript
    def on_transcript(event):
        if event.is_final:
            print(f"[{event.type}] {event.text}")

    await translator.start(ctx.room)
```

---

## Voice types

| Value | Description |
|-------|-------------|
| `clone` | Preserves the speaker's original voice identity (default) |
| `female` | Standard female voice |
| `male` | Standard male voice |

---

## Supported languages

Full list of language codes: [supported languages](https://www.startpinch.com/docs/supported-languages)

---

## License

Apache 2.0 — see [LICENSE](./LICENSE) for details.
