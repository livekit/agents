# LemonSlice Avatar with Switchable Personas

A voice agent paired with a [LemonSlice](https://www.lemonslice.com/) animated
avatar. The agent boots into one of 15 hard-coded personas — each pairs an
image (the avatar that LemonSlice renders), a Cartesia voice id, a system
prompt, and a pair of body-language prompts (speaking / idle). The playground
dropdown picks the persona by id and dispatches the choice as agent metadata;
the agent resolves it and wires up the right TTS + prompt.

## Layout

```
examples/avatar/
  agent.py        # AgentSession, the set_avatar RPC, the hold-music
                  # context manager.
  personas.py     # `Persona` dataclass + the `PERSONAS` library +
                  # `resolve_persona` helper + the COMMON_INSTRUCTIONS
                  # shared directive that's appended to every persona's
                  # system prompt.
  hold_music.py   # Procedural F-major pairing-tone-style synth used as
                  # background hold music while a persona swap is in
                  # flight. No audio file shipped.
  livekit.toml    # Regenerated from playground.yaml by the deploy
                  # workflow; committed for local-dev convenience.
  Dockerfile, requirements.txt
```

## Personas

Defined in `personas.py:PERSONAS`. The set: `influencer` (default),
`software_engineer`, `music_teacher`, `social_worker`, `joyce`, `iris`,
`ai_therapist`, `management_consultant`, `shopping_assistant`, `cat_girl`,
`mock_interviewer_legal`, `mr_fox`, `monroe`, `fortnite_guide`, `kitten_tutor`.

Each `Persona` carries `image_url`, `voice_id`, `system_prompt`,
`speaking_prompt`, and `idle_prompt`. The body-language prompts are forwarded
to LemonSlice as `agent_prompt` / `agent_idle_prompt`.

The shared `COMMON_INSTRUCTIONS` block in `personas.py` is appended to every
persona's system prompt — that's where the global "speak naturally, one or
two short sentences, no markdown" rules live.

## Switching personas at runtime

The `set_avatar` RPC accepts `{"value": "<persona_id>"}` and:

1. Plays a short pairing-tone hold loop on a background audio track.
2. Closes the previous LemonSlice `AvatarSession`.
   `BaseAvatarSession.aclose()` kicks the avatar participant via
   `RoomService.RemoveParticipant` so its audio + video publications are
   torn down.
3. Starts a fresh `AvatarSession` with the new image, `agent_prompt`,
   and `agent_idle_prompt`. `wait_for_join()` blocks until the new
   participant + video track are present.
4. `session.update_agent(make_agent(new_persona))` swaps in the new TTS
   voice + system prompt.
5. `session.generate_reply(...)` triggers the new persona to introduce
   itself; the hold music fades out as that audio starts.

Concurrent `set_avatar` calls are serialised behind an `asyncio.Lock`; a
call that arrives while a swap is in flight is rejected with an RPC
application error so the frontend can surface a "try again in a moment"
toast.

The RPC returns `{"id": "<persona_id>"}`. The image swap happens fully
mid-session — no client reconnect required.

## Initial persona on connect

The agent reads the persona id from `ctx.job.metadata`. The playground sends
either a JSON `{"persona_id": "..."}` object or a bare id string; both are
accepted. Unknown or missing ids fall back to `DEFAULT_PERSONA_ID = "influencer"`.

## Running locally

```bash
export LEMONSLICE_API_KEY="sk_lemon_..."
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
export LIVEKIT_URL="..."

python examples/avatar/agent.py dev
```

The example also calls `load_dotenv(find_dotenv(...))`, so any `.env` walking
up from the file is picked up.

## Deployment

`playground.yaml` carries the avatar entry (pixel icon, accent, persona
dropdown, `agent_id`). The CI workflow at
`.github/workflows/deploy-examples.yml` includes `avatar` in its matrix and
regenerates `livekit.toml` from `playground.yaml` before invoking
`lk agent deploy`.

Per-agent secrets are set out of band, once:

```bash
cd examples/avatar
lk agent update-secrets --project examples \
  --secrets "LIVEKIT_AGENT_NAME=avatar" \
  --secrets "LEMONSLICE_API_KEY=$LEMONSLICE_API_KEY"
```
