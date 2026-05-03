---
"livekit-agents": minor
---

Add `--reload-dir` (repeatable) and `LIVEKIT_RELOAD_DIRS` to `dev` so the watcher
restarts the worker on edits inside additional, user-supplied directories.
Auto-discovery of the entrypoint, `livekit.agents`, and editable plugins is
unchanged; the new flag is purely additive.
