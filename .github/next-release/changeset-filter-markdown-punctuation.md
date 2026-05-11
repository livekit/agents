---
"livekit-agents": patch
---

Fix `filter_markdown` to strip `**bold**` and `*italic*` when the delimiters are adjacent to punctuation (e.g. `**name**!`, `**date**,`, `(**1**)`). The previous regexes used `(?<!\S)` / `(?!\S)` boundaries which rejected punctuation as a valid boundary, letting raw Markdown leak through to TTS.
