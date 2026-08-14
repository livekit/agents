# Microsoft Teams plugin by StandIn for LiveKit Agents

Answer Teams calls and Teams chat messages with a LiveKit agent.

[StandIn](https://standin.komaa.com) is the hosted service that joins the Microsoft Teams call; this plugin answers StandIn's per-call dial inside your agent worker. It is standalone: one worker process creates one LiveKit room per call in your own project, dispatches your agent into it by name, and relays the audio both ways. There is no separate bridge to run and nothing to deploy alongside.

See [docs.komaa.com/livekit/installation](https://docs.komaa.com/livekit/installation) for the full setup.

## Installation

```bash
pip install livekit-plugins-standin
```

## Pre-requisites

You'll need a connection secret from StandIn. It can be set as an environment variable: `STANDIN_SECRET`. Your LiveKit project is read from the standard `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`.

## Tiers

Sandbox, Free, Paid and Managed all run the same code. The tier belongs to the connection you paired in the portal, so there is nothing to select here. Managed connections additionally carry the messages lane described below.

## Calls

Your file is shaped like every other agent example, and nothing starts except through `cli.run_app(server)`:

```python
from livekit.plugins import standin

server = AgentServer()


@server.rtc_session(agent_name="msteams-agent")
async def entrypoint(ctx: JobContext):
    session = AgentSession(llm=...)
    call = await standin.TeamsCall().start(session, ctx=ctx)
    await session.start(agent=MyAgent(call), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
```

There is no bootstrap call. Importing the plugin registers it with the worker, and setting `STANDIN_SECRET` arms it: the call listener starts with the worker on the agreed layout, port **8080** at **`/msteams/calling`** (`STANDIN_PORT` / `STANDIN_WS_PATH` override), and stops with it. A worker without `STANDIN_SECRET` behaves as if the plugin were not installed. The agent name is read off `@server.rtc_session(agent_name=...)`, so it is declared exactly once.

Expose the port and register the public URL as the identity's **Agent voice URL** in the portal, for example:

```bash
tailscale funnel --bg --set-path /msteams/calling http://127.0.0.1:8080/msteams/calling
```

Per call, the plugin verifies StandIn's HMAC handshake, creates room `msteams-{callId}`, dispatches your agent with the call metadata, publishes the caller's audio as a room track, and relays your agent's audio back to Teams. In the entrypoint, `TeamsCall().start()` reads `CallInfo` (`caller_name`, `tenant_id`, `call_id`, `thread_id`, `user_id`, `direction`) and handles the two data topics:

- `msteams.context` - non-interrupting context text: participant changes, DTMF digits, recording state. Pass `on_context=` to read it; by default it is only logged.
- `msteams.goodbye` - StandIn is ending the call and wants this line spoken first. The default handler interrupts the current turn and says it; pass `on_goodbye=` to replace that.

`CallInfo.from_job(ctx).is_teams_call` is False for a job dispatched by anything else, so one worker can serve Teams, SIP and web rooms at once.

## Messages

Managed connections only, which needs no flag: the channel authenticates with your connection secret, so it opens only for a live managed connection.

```python
async def on_message(msg: standin.InboundMessage) -> str:
    return f"You said: {msg.text}"


chat = standin.ChatChannel(respond=on_message)
await chat.start()
```

The worker dials **out** to StandIn; messages are pushed down that socket and replies go back up it, so this lane needs no port. StandIn authenticates the Teams activity, strips the bot @mention, and performs the Teams send - your agent never holds a Bot Framework credential.

Turns are serialized per conversation, deduped on `activityId`, bounded by a turn timeout, and answered with a typing indicator while your handler thinks.

## Example

A complete single-file worker: [`examples/msteams`](../../examples/msteams/).

## Roadmap

- **Avatar** - the video surface (the agent's face on the Teams tile, caller screen-share and camera) depends on a third-party avatar runtime, so it lands separately. Receiving one of those frames is not an error; it is ignored like any unknown message type.

## License

Apache-2.0
