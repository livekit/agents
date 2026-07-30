# Homepage knowledge agent

A voice agent that answers questions about LiveKit products. Knowledge about the
Agents SDKs stays in the system prompt for low-latency answers, while other product
knowledge is loaded on demand through a generated `lookup_product` tool.

## Architecture

- `agent.py` is the composition root. It contains the immutable `AgentConfig`,
  `Assistant`, session setup, and server entrypoint.
- `knowledge_base/` discovers one Markdown file per product and derives both the
  tool schema and lookup results from those files.
- `prompts/` contains all authored agent language as Markdown templates.
- `behaviors/` contains session event behavior and frontend integration.
- `filters/` contains streaming voice-pipeline transformations.
- `tests/unit/` is deterministic; `tests/evals/` runs live behavioral evaluations.

The voice pipeline uses LiveKit Inference with Gemma 4 31B, Deepgram Nova-3,
Inworld TTS, the LiveKit turn detector, and ai-coustics voice isolation.

## Run locally

From this directory, install the example dependencies and provide LiveKit Cloud
credentials in `../.env` or the environment:

```bash
pip install -r requirements.txt
python agent.py console
```

Use `python agent.py dev` to connect the agent to LiveKit Cloud for a frontend or
telephony session.

## Tests and evals

Install the repository's development dependencies, then run the fast unit suite:

```bash
python -m pytest
```

Run the live LLM-backed eval suite with:

```bash
python -m pytest -m evals
```

The evals require `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` and cover tool routing,
grounded facts, anti-hallucination behavior, inline knowledge, and multi-turn grounding.
