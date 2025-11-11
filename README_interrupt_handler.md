\# LiveKit Interrupt Handler 



\##  Overview

This feature adds an \*\*intelligent interruption filter\*\* for LiveKit voice agents.  

It prevents false interruptions caused by filler words such as “uh”, “umm”, “hmm”, or “haan” while the agent is speaking.



When genuine speech or commands like “wait” or “stop” occur, the agent stops immediately, maintaining natural, smooth conversations.





\## Objective

\- Ignore filler words \*\*only when the agent is speaking\*\*

\- Detect real user interruptions instantly

\- Keep LiveKit’s base Voice Activity Detection (VAD) unchanged

\- Work asynchronously for real-time response

\- Support dynamic configuration via environment variables





\##  What Changed

\- Added a new module: `interrupt\_handler.py`

\- (Optional demo) Created `main.py` to simulate agent behavior using the new handler





\## What Works

| Scenario | Transcript | Agent Speaking | Result |

|-----------|-------------|----------------|---------|

| Filler only | “umm”, “hmm”, “haan” | Ignored |

| Real command | “stop”, “wait” | Agent stops immediately |

| Mixed speech | “umm okay stop” | Agent stops |

| Filler only | “umm” |Registered as normal speech |



---



\## Configuration

You can customize filler and command words dynamically.



| Environment Variable | Description | Default |

|----------------------|--------------|----------|

| `IGNORED\_WORDS` | Comma-separated list of fillers | `uh,umm,hmm,haan` |

| `COMMAND\_WORDS` | Comma-separated list of interrupt commands | `stop,wait,pause,hold` |

| `MIN\_ASR\_CONFIDENCE` | Minimum ASR confidence (0–1) | `0.6` |



Example:

```bash

setx IGNORED\_WORDS "uh,umm,hmm,haan,achha"

setx COMMAND\_WORDS "stop,wait,pause,hold"

setx MIN\_ASR\_CONFIDENCE "0.7"



