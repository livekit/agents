# Homepage knowledge agent

The voice agent on livekit.io's homepage: it answers questions about LiveKit products.
Knowledge about the Agents SDKs stays in its instructions for low-latency answers; every other
product is loaded on demand through a `lookup_product` tool.

| File | What |
| --- | --- |
| `main.py` | Models, the agent and its greeting, the session and the server |
| `instructions.md` | The agent's instructions, with the inline Agents SDK knowledge |
| `knowledge.py`, `products/` | `lookup_product`: one Markdown file per product, whose first line is its index entry |
| `behaviors.py` | Checking in when the caller goes quiet, hanging up on scripted callers, the TTS voice attribute for the homepage panel, saying "LiveKit" right |
| `scripted_questions.txt` | Opening lines of the automated callers that are hung up on |
| `tests/` | `test_unit.py` is deterministic; `test_evals.py` runs the agent and an LLM judge live |

The voice pipeline uses LiveKit Inference with Gemma 4 31B, Deepgram Nova-3, Fish Audio S2.1 Pro
in expressive mode, the LiveKit turn detector and ai-coustics voice isolation.

## Run locally

In this repository the example is a uv workspace member: it runs on the in-repo SDK, in the
root `.venv` (`make install` at the root). Provide LiveKit Cloud credentials in `.env` or the
environment, then from this directory:

```bash
lk agent console main.py              # talk to it in the terminal
lk agent dev main.py                  # connect it to LiveKit Cloud
lk agent simulate -n 5 text main.py   # run generated, judged scenarios
```

## Tests and evals

```bash
uv run pytest              # the unit tests
uv run pytest -m evals     # the live evals; needs LIVEKIT_API_KEY and LIVEKIT_API_SECRET
```

The evals cover tool routing, grounded facts, anti-hallucination, inline knowledge, multi-turn
grounding and conversation basics. They call live models, so re-run a lone failure before
treating it as a regression.

## Deploy

```bash
lk agent deploy .
```

The deploy builds this directory alone with the `Dockerfile`: `pyproject.toml` resolves
`livekit-agents` and the plugins from PyPI, not from this repository, so the deployed agent
runs on the latest published release. The agent registers as `homepage_agent_v3`.
