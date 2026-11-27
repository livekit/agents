# Rabel plugin for LiveKit Agents

Starts your voice agent's reply early on [Rabel](https://rabellabs.com) Bell's guess at the caller's final words. Bell is
a hosted model that learns each agent's callers from that agent's finished calls. The plugin wraps your streaming
speech-to-text and sends Bell's guess to LiveKit as a preflight transcript, so preemptive generation starts the LLM on
it. LiveKit keeps that reply only when the final transcript matches. Your speech-to-text still ends the turn.

More information: [rabellabs.com/docs](https://rabellabs.com/docs).

## Installation

```bash
pip install "livekit-agents[rabel]"
```

## Pre-requisites

A Rabel API key, set as the environment variable `RABEL_API_KEY`.

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import deepgram, rabel

session = AgentSession(
    stt=rabel.STT(deepgram.STTv2(model="flux-general-en"), agent="front-desk"),
    turn_handling={"preemptive_generation": {"max_retries": 4}},
    # llm, tts and vad as usual
)
```

Preemptive generation must be on (LiveKit's default). The plugin only sends a guess while LiveKit keeps two early starts
of its own after it, counted as LiveKit counts them, so a guess does not make a reply start later. With `max_retries` 4
Bell can guess twice in a turn, at the default of 3 once, and with 2 or fewer not at all. Where Bell has no guess, the
speech-to-text's events pass unchanged. If the API is slow or down, the plugin fails open.

A new agent sends no guess until Bell has learned from enough of its calls. `agent="demo-bank"` is a ready-made agent
trained on a bank's calls, for trying the plugin at once.

## Options

| Option | Default | |
|---|---|---|
| `agent` | `"default"` | Your agent's name. Calls under one name teach one Bell. |
| `min_chance` | `None` | How sure Bell must be of a guess, from `0.67` to `0.99`. `None` uses Bell's balanced setting. |
| `answer_timeout` | `0.5` | Seconds after its words within which a guess is still used. |
| `learn` | `True` | `False`: the calls through this wrapper teach Bell nothing. |
| `report` | `True` | Tells the API at each turn's end whether the guess was kept, with no words. |
| `enabled` | `True` | `False` makes the wrapper a plain pass-through. |
| `api_key` | `RABEL_API_KEY` | Your Rabel API key. |
| `base_url` | `RABEL_API_URL`, else `https://api.rabellabs.com` | The API's address. |

## Limits

- While a guess is live, interim transcripts that agree with it are held back, so a live caption shows the guess.
- With Deepgram Flux's eager end of turn on, Bell sends no guess after Flux's own early start in a turn.
- An app that keeps turn detection automatic and also calls `session.commit_user_turn()` itself: on LiveKit versions
  where a commit within half a second of a preflight does not wait for the final transcript, that commit can take a
  guess as the caller's words. Such an app sets `enabled=False`.
