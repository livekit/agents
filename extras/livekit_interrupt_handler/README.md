\# LiveKit Interrupt Handler — Filler-Aware Extension



\##  What Changed

Added an extension module under `extras/livekit\_interrupt\_handler/` to filter filler utterances like "uh", "umm", "hmm", and "haan" while the LiveKit agent is speaking.



\##  Features

\- Suppresses filler-only utterances during TTS.

\- Forwards real interruptions (e.g. "wait", "stop") instantly.

\- Registers fillers as valid speech when the agent is quiet.

\- Supports runtime filler updates via FastAPI API.

\- Fully async and thread-safe; no modification to LiveKit base SDK.



\##  Files Added

| File | Purpose |

|------|----------|

| `src/filler\_filter.py` | Core filtering logic |

| `src/integration\_demo.py` | Demonstration of the filter behavior |

| `src/runtime\_config\_api.py` | FastAPI server for runtime config |

| `tests/test\_filler\_filter.py` | Unit tests using pytest |



\##  How to Test

1\. Install dependencies:

&nbsp;  ```bash

&nbsp;  pip install fastapi uvicorn pytest



