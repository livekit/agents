# Feedback interview

The feedback interview from [livekit.com/feedback](https://livekit.com/feedback). Morgan, an AI
interviewer, asks someone who evaluated LiveKit how it went: who they are, how they first heard
about LiveKit, who else they considered, and what got in the way. It takes about five minutes,
and Morgan digs for specifics rather than selling.

## The calls

Someone starts the interview from the page, or calls the agent's LiveKit Phone Number. Morgan says
the call is recorded, then interviews them.

## How it's built

- `src/agent.py` is the entrypoint for the `homepage-feedback-interview` agent. It sets up the
  session and starts the call.
- `src/interviewer.py` holds Morgan's role, `Interviewer`: the instructions, the docs lookup, and
  the interview clock.
- `src/call_summary.py` keeps a summary of the interview on the agent participant for the page:
  see [What the page reads](#what-the-page-reads).
- `src/playout_hold.py` holds the first 100 ms of each burst of Morgan's audio, so GPT-Live's
  delivery jitter can't run playback dry, which you hear as a click. It patches livekit-rtc until
  the SDK buffers realtime playback itself.

The agent is one speech-to-speech session on GPT-Live 1.1 (`openai/gpt-live-1.1`, voice Marin)
through LiveKit Inference, with no separate speech-to-text or text-to-speech. Four things shape the
interview:

- **The instructions** in `INSTRUCTIONS` say what to learn and how hard to dig. Edit them to change
  what Morgan asks.
- **The docs lookup** answers product questions from the
  [LiveKit Docs MCP server](https://docs.livekit.io/reference/developer-tools/docs-mcp/), never
  from memory. Only `docs_search` and `get_pages` are exposed, and each result is cut to 4,000
  characters so the next reply stays quick.
- **The interview clock** appends the elapsed and remaining time to GPT-Live's thinking channel
  every minute, so Morgan paces to five minutes without mentioning the clock. A minute past time,
  an instruction to wrap up follows.

- **The hang-up** is decided by a decision model (`openai/gpt-6-luna-decisions`), which reads each
  of Morgan's turns and ends the call once one is a farewell. GPT-Live rarely delegates a hang-up
  on its own.

ai-coustics cleans up the caller's audio, and a VAD lets the caller cut Morgan's playback short.

## What the page reads

The agent publishes one participant attribute, `interview.summary`, as a JSON string:

```json
{
  "name": "Shane",
  "role": null,
  "summary": "- They were building an AI to answer the phone for their medical practice.\n- They found LiveKit by Googling medical AI and seeing a demo.\n- They chose Vapi because it had a Jane connector."
}
```

- `name` and `role` are what the interviewee said, or `null` if they didn't say.
- `summary` is Markdown: a bulleted list, one `- ` line per point, usually three to six. The model
  returns plain points and the agent builds the list, so it's always a valid list.

Gemma 4 31B (`google/gemma-4-31b-it`) rewrites it after each of Morgan's turns, so the page has the
latest one even when the visitor hangs up first, and once more before Morgan ends the call. It isn't
set until the interviewee has spoken.

## Run it locally

```console
uv sync
lk app env --write --destination .env.local
lk agent dev
```

Then talk to it in the [Agent Console](https://docs.livekit.io/testing/agent-console/), or run
`lk agent console` to talk to it in your terminal. To try it one spoken turn at a time, use the
[agent debugger](https://docs.livekit.io/testing/debugger/) in audio mode, since GPT-Live only
takes audio:

```console
lk agent debugger start --audio
lk agent debugger say "I'm Sam, a founder. We're building a support line for a bank."
lk agent debugger stop
```

## Tests and simulations

```console
uv run pytest
lk agent simulate audio --scenarios scenarios.yaml
```

The scenarios run in audio for the same reason, and are judged on each one's
`agent_expectations`.

## Deploy

```console
lk agent deploy .
```

This needs a `livekit.toml` that names the agent. It's gitignored, and the repository's
[`Deploy Homepage Examples`](../../../.github/workflows/deploy-homepage-examples.yml) workflow
writes one for each homepage demo.
