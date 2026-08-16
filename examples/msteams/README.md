# Microsoft Teams plugin by StandIn for LiveKit Agents

A LiveKit agent that answers **Microsoft Teams** calls, through [StandIn](https://standin.komaa.com).

One file, one process, run like any other agent worker:

```bash
uv sync
uv run python -m livekit.agents download-files 
uv run agent.py dev
```

## Setup

```bash
STANDIN_SECRET=...            # the connection secret from the StandIn portal
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

Put those in `.env`, run the worker, then expose its port (9442 by default) with a tunnel or ingress, for example `tailscale funnel --bg --set-path /msteams/calling http://127.0.0.1:9442/msteams/calling`, and register the public URL in the StandIn portal as the agent voice URL: `wss://<your-host>/msteams/calling`.

## What the agent gets

`CallInfo` carries `call_id`, `thread_id`, `caller_name`, `user_id` (the caller's AAD id when Teams provides one), `tenant_id` and `direction`. `call.is_teams_call` is False when the job came from somewhere else, so one worker can serve Teams, SIP and web rooms at the same time.

Two data topics, both handled in `worker.py`:

- `standin.TOPIC_CONTEXT` - participant count, Teams recording state, DTMF digits
- `standin.TOPIC_GOODBYE` - StandIn's governor is ending the call and wants the agent to say this first, so interrupt the current turn to deliver it

