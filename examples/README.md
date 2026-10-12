# Fast brain, slow brain

Three examples of the pattern: a front desk that talks to the person, by text and on the
phone, and knows nothing but whom to ask; a back office that holds the system and does the
work, reached over A2A as the front desk's only tool.

| Example                       | What                                                                                                  |
| ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| [`frontdesk/`](frontdesk/README.md) | A dental practice's appointment scheduling on a calendar: the pattern at its smallest, two tools and a booking that asks for what it is missing. |
| [`hotel-receptionist/`](hotel-receptionist/README.md) | A hotel's bookings, lookups and cancellations: guided flows as tasks awaited in text, verification, approval on the writes. |
| [`airline-support/`](airline-support/README.md) | An airline's customer care: one agent whose tool awaits a graph that asks for what is missing and continues on the answer, EU261 in code, a reviewed reply, payment only after approval. |

Both write their own lane of one session: the back office keeps the person's details on its
lane, so the front desk carries no state, and the conversation's history shows the back
office's tasks beside the front desk's turns. The two run on one worker: each agent file
defines an `AgentHandler` with its doors, the back office's on a lane of its own, and
`main.py` makes the server and adds both. A deployment of its own is the same `backend_agent.py` behind its own `main.py`. The console (`../console`) lists them all and
talks to whichever is running.

## Environment

The workers need a LiveKit project, the agent store that holds the threads, and the models:

```sh
export LIVEKIT_URL=<project URL>                 # wss://<project>.staging.livekit.cloud
export LIVEKIT_API_KEY=<key>
export LIVEKIT_API_SECRET=<secret>
export LIVEKIT_AGENTDB_URL=<agent store data plane>      # ws://.../v1/db
export LIVEKIT_AGENT_THREADS_URL=<threads service>        # the AgentThreads gRPC service
export OPENAI_API_KEY=<key>                      # the phone door's realtime model
export LIVEKIT_INFERENCE_API_KEY=<key>           # the text agents' model, through LiveKit inference
export LIVEKIT_INFERENCE_API_SECRET=<secret>
```

The agent store takes the LiveKit key pair unless `LIVEKIT_AGENTDB_API_KEY` and
`LIVEKIT_AGENTDB_API_SECRET` name its own. Once, from the repository root: `uv sync --all-packages`.

<details>
<summary>A local stack, for LiveKit engineers</summary>

- livekit-server at `ws://127.0.0.1:7880` (key `devkey`), WebTransport on `:7883`; endpoints
  are at `http://127.0.0.1:7880/agents/<agent>/default/<endpoint>`
- agent-db: data plane `ws://localhost:7781/v1/db`, streams `http://localhost:7780`, management
  gRPC `http://localhost:7783`, threads gRPC `http://localhost:7793`; the livekit-server's key
  pair, so the workers' one credential reaches every service

The configs are agent-db's, in its `dev/stack/`, beside the `dev/docker-compose.yaml` that
brings up the Scylla and MinIO it wants; the server build must carry the agent endpoints and
`agents.endpoints.threads_address`.

```sh
docker compose -f <agent-db>/dev/docker-compose.yaml up -d --wait
<agent-db>/bin/agentstore --config <agent-db>/dev/stack/agentstore.yaml db
<agent-db>/bin/agentstore --config <agent-db>/dev/stack/agentstore.yaml threads
livekit-server --config <agent-db>/dev/stack/livekit.yaml
```

```sh
export LIVEKIT_URL=ws://127.0.0.1:7880
export LIVEKIT_API_KEY=devkey
export LIVEKIT_API_SECRET=secretsecretsecretsecretsecretsecret
export LIVEKIT_WORKER_URL=https://127.0.0.1:7883
export LIVEKIT_WORKER_INSECURE=1
export LIVEKIT_AGENTDB_URL=ws://localhost:7781/v1/db
export LIVEKIT_AGENTDB_STREAMS_URL=http://localhost:7780
export LIVEKIT_AGENTDB_MANAGEMENT_URL=http://localhost:7783
export LIVEKIT_AGENT_THREADS_URL=http://localhost:7793
```

</details>

## Run

The console (`../console`) is the way in: its API and web app, then pick an example in it;
one that is not running shows its start command and connects once it is up. From an example's
directory, `main.py` runs both agents on one worker:

```sh
(cd ../../console && uv run python console_api.py)           # the console's API on :7795
(cd ../../console && pnpm install && pnpm dev)               # http://localhost:3000/
uv run python main.py dev
```

Pick the example in the console, type a request, or press the speech button to call the front
desk into the same conversation. Each delegation shows as a card in the thread, and the debug
drawer under it shows the thread's lanes, tasks, journal and inbox as they change.

## Tests

`tests/test_v0_checks.py` runs the plan's text checks against this stack (it skips when the
stack is down); the console's e2e makes the call.

```sh
V0_STACK=1 uv run pytest tests/test_v0_checks.py -v
```
