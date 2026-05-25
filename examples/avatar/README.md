# LemonSlice Avatar with Switchable Personas

A voice agent paired with a [LemonSlice](https://www.lemonslice.com/) animated
avatar. The agent boots into one of seven hardcoded personas — each persona
binds an image (rendered by LemonSlice), a Cartesia voice id, and a system
prompt. The frontend playground exposes a dropdown that picks the persona by
id; the agent does the resolution.

## Personas

Defined in `agent.py` under `PERSONAS`. Currently:

- `influencer`
- `software_engineer`
- `music_teacher`
- `spanish_tutor`
- `social_worker`
- `joyce`
- `iris`

Each persona has `image_url`, `voice_id`, and `system_prompt` fields. Tune
those (especially the voice ids — the defaults are placeholders matching the
Cartesia sonic-3 library) without touching anything on the frontend.

## Switching personas at runtime

The `set_avatar` RPC accepts `{"value": "<persona_id>"}` and swaps the TTS
voice + system prompt live. The LemonSlice video stream is bound at
`AvatarSession.start()` time and cannot be re-pointed, so a full visual swap
also needs the client to reconnect with a new persona id in the dispatch
metadata.

The RPC's response is JSON-encoded with two fields:

```json
{ "id": "spanish_tutor", "reconnect_required": true }
```

`reconnect_required` is `true` whenever the persona id actually changed —
this lets the frontend surface a "reconnect to update the avatar visual" hint
if it chooses.

## Running

```bash
export LEMONSLICE_API_KEY="..."
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
export LIVEKIT_URL="..."

python examples/avatar/agent.py dev
```

No `LEMONSLICE_IMAGE_URL` env var is required — every persona ships with its
image url hardcoded in `agent.py`.
