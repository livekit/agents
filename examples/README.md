# Fast brain, slow brain

A front desk talks to the person, by text and on the phone; its one tool delegates over A2A to a
back office that holds the system and does the work.

| Example | What |
| --- | --- |
| [`frontdesk/`](frontdesk/README.md) | A dental practice's scheduling: two tools and a booking that asks for what it is missing. |
| [`hotel-receptionist/`](hotel-receptionist/README.md) | A hotel's bookings, lookups and cancellations: guided flows, verification, approval on writes. |
| [`airline-support/`](airline-support/README.md) | An airline's customer care: a graph that asks for what is missing, EU261 in code, payment after approval. |

## Quickstart

You need a LiveKit Cloud staging project: copy its URL, API key and secret from the dashboard.
Threads are stored in the project with the same key.

```sh
uv sync --all-packages                            # once, from the repository root

export LIVEKIT_URL=wss://<project>.staging.livekit.cloud
export LIVEKIT_API_KEY=<key>
export LIVEKIT_API_SECRET=<secret>
export OPENAI_API_KEY=<key>                       # calls only: the realtime model
export LIVEKIT_WORKER_INSECURE=1                  # temporary: staging's QUIC certificate
export LIVEKIT_AGENT_DEPLOYMENT=default           # temporary: staging's endpoint routing

cd examples/frontdesk && uv run python main.py dev
```

Then open the console in two more terminals, with the same environment, and pick the example:

```sh
cd console && uv run python console_api.py
cd console && pnpm install && pnpm dev            # http://localhost:3000/
```

Type a request, or press the speech button to call the front desk into the same conversation.
