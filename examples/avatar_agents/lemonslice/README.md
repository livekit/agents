# LiveKit LemonSlice Avatar Agent

This example demonstrates how to create an animated avatar using [LemonSlice](https://www.lemonslice.com/).

## Usage

* Update the environment:

```bash
# LemonSlice Config
export LEMONSLICE_API_KEY="..."
export LEMONSLICE_IMAGE_URL="..." # Publicly accessible image url for the avatar.

# LiveKit config
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
export LIVEKIT_URL="..."
```

* Start the agent worker:

```bash
python examples/avatar_agents/lemonslice/agent_worker.py dev
```

## Third-party video meeting platforms (Zoom, Meet, Teams, Webex)

Use `agent_worker_meeting.py` to send the avatar into a third-party video meeting.
The avatar joins the call, listens to meeting audio, and responds through the meeting
relay.

Set the meeting URL via job metadata when dispatching the agent. For password-protected
meetings, include the password in the URL (for example, Zoom links use a `pwd` query
parameter):

```json
{
  "meeting_url": "https://zoom.us/j/123456789?pwd=abcdef",
  "bot_name": "LemonSlice Avatar",
  "listen_to_meeting_chat": true
}
```

For local testing, you can also set `MEETING_URL` (and optionally `MEETING_BOT_NAME`
or `LISTEN_TO_MEETING_CHAT`) in the environment instead of job metadata.
`LISTEN_TO_MEETING_CHAT` accepts `true`/`false` or `1`/`0`.

```bash
export MEETING_URL="https://zoom.us/j/123456789?pwd=abcdef"
export LISTEN_TO_MEETING_CHAT="false"
python examples/avatar_agents/lemonslice/agent_worker_meeting.py dev
```
