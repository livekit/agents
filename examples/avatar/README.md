# LemonSlice Avatar

A voice agent with a talking-head avatar you can swap mid-conversation.
Pick a persona from the dropdown — Leila, Jess, a software engineer, a
cat, a fox — and the agent's face, voice, and personality
all change without dropping the call.

Try it in the [LiveKit Playground](https://agents.livekit.io/?example=avatar).

## What's in here

- **9 personas** to choose from — each has its own face, voice, system
  prompt, and idle/speaking body-language hints.
- **Live persona switching** — the dropdown fires a `set_avatar` RPC; a
  short hold tone plays while the avatar reconnects with the new face
  and voice.
- **Hero motions** — for **Leila**, **Jess**, and **Mr Fox**, the LLM can
  trigger wave, dance, or turn via tool calls (one motion at a time,
  ~6 seconds each). They wave automatically when the session starts.
- **LiveKit Inference** for STT + LLM (Deepgram Nova-3 + Gemini 3.5
  Flash), Cartesia for TTS, [LemonSlice](https://lemonslice.com) for
  the avatar video.

## Running it locally

You'll need:

- A LiveKit Cloud project (`LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`,
  `LIVEKIT_URL`).
- A LemonSlice API key — get one from
  [lemonslice.com](https://lemonslice.com). Export it as
  `LEMONSLICE_API_KEY`.

Then:

```bash
pip install -r requirements.txt
python agent.py dev
```

Connect from any LiveKit client. The agent reads the starting persona
from the job metadata; if no metadata is sent it defaults to Leila.

## Adding or editing personas

Everything lives in [`personas.py`](./personas.py). Each entry has:

- `image_url` — the picture LemonSlice will animate.
- `voice_id` — a Cartesia voice id.
- `system_prompt` — who the persona *is*. Keep it tight; the shared
  `COMMON_INSTRUCTIONS` block already covers global rules (be brief,
  no markdown, etc.).
- `speaking_prompt` / `idle_prompt` — short body-language cues sent to
  LemonSlice (`agent_prompt` / `agent_idle_prompt`).

To add a persona, append a new `Persona(...)` to the `PERSONAS` dict.
The playground UI auto-discovers it once
[`examples/playground.yaml`](../playground.yaml) is updated too.

## How persona switching works

When the playground dropdown changes, the frontend calls the agent's
`set_avatar` RPC. The agent:

1. Plays a short hold tone in the background.
2. Closes the current avatar session.
3. Opens a fresh one with the new face + body-language prompts.
4. Swaps the TTS voice + system prompt.
5. Greets you as the new persona.

No reconnect, no page refresh — the same call, with a different face.

## Files

```
agent.py        entry point + the set_avatar RPC
actions.py      pose controller (opening wave + LLM tool motions)
personas.py     the 9 personas and the shared prompt rules
hold_music.py   the soft three-note "please wait" tone
Dockerfile      for cloud deploys
```
