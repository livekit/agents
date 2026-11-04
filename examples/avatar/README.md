# Avatar

A voice agent with a talking-head avatar you can swap mid-conversation.
Pick a persona from the dropdown — Leila, Jess, a software engineer, a
cat, a fox, or a Tavus face — and the agent's face, voice, and personality
all change without dropping the call. Each persona picks its avatar
provider (`provider="lemonslice"` or `provider="tavus"` in
[`personas.py`](./personas.py)), so you can switch providers mid-call too.

Try it in the [LiveKit Playground](https://agents.livekit.io/?example=avatar).

> **Inference variant:** [`inference_agent.py`](./inference_agent.py) is a
> minimal version that provisions the avatar through **LiveKit Inference**
> instead of the BYOK plugin — no `LEMONSLICE_API_KEY` is needed; the agent
> authenticates with your LiveKit credentials and the gateway creates the
> provider session with LiveKit's wholesale key. Requires the
> `avatar_lemonslice` feature flag on your project. Run with
> `uv run inference_agent.py dev` and set `LIVEKIT_URL`, `LIVEKIT_API_KEY`,
> `LIVEKIT_API_SECRET`, and `LEMONSLICE_IMAGE_URL`.

## What's in here

- **9 personas** to choose from — each has its own face, voice, system
  prompt, and idle/speaking body-language hints.
- **Live persona switching** — the dropdown fires a `set_avatar` RPC; a
  short hold tone plays while the avatar reconnects with the new face
  and voice.
- **Hero motions** — for **Leila**, **Jess**, and **Mr Fox**, the LLM can
  trigger wave, dance, or turn via tool calls (one motion at a time,
  ~6 seconds each). They wave automatically when the session starts.
- **Facial emotions** — for the **Tavus** persona, the LLM sets the
  face's emotion (happiness, sadness, surprise, …) via a `set_emotion`
  tool call as the mood of the conversation changes.
- **LiveKit Inference** for STT + LLM (Deepgram Nova-3 + Gemini 3.5
  Flash), Cartesia for TTS, [LemonSlice](https://lemonslice.com) or
  [Tavus](https://tavus.io) for the avatar video.

## Running it locally

You'll need:

- A LiveKit Cloud project (`LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`,
  `LIVEKIT_URL`).
- A LemonSlice API key — get one from
  [lemonslice.com](https://lemonslice.com). Export it as
  `LEMONSLICE_API_KEY`.
- For the Tavus persona, a Tavus API key exported as `TAVUS_API_KEY`.
  Optionally set `TAVUS_FACE_ID` / `TAVUS_PAL_ID` (or `face_id` on the
  persona); otherwise the stock pal is used.

Then:

```bash
uv sync --all-extras --dev   # from the repository root
uv run agent.py dev
```

Connect from any LiveKit client. The agent reads the starting persona
from the job metadata (`{"set_avatar": "tavus"}` starts with Tavus);
if no metadata is sent it defaults to Leila.

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
