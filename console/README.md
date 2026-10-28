# Console

A local developer console for talking to an agent by text or voice. It uses the
threads service for conversation lists and history, and authenticated A2A 1.0
JSON-RPC POST streams for replies and task replay. Every agent card and endpoint
request carries a scoped bearer token. Project credentials stay on the backend.

Run the example's workers and LiveKit services first. Set the variables in
`console/.env.example` in both the Python backend and Next.js environments; use a
random `CONSOLE_API_TOKEN` shared by those two processes. Next.js reads
`console/.env.local`; the Python process reads its environment. From the repository root:

```sh
uv run python console/console_api.py
pnpm --dir console dev
```

Open `http://localhost:3000`. Both console servers bind to loopback. The Next.js
routes accept only `CONSOLE_ORIGIN` (default `http://localhost:3000`), reject
cross-origin requests, and authenticate to the Python backend using the shared
secret. This is a local developer tool; it has no hosted-user login flow.

The example picker discovers `examples/*/console.toml`. Conversations are listed
by subject and can be continued by any configured agent. Text first contact
learns the thread and task from the acknowledgment. Reloading after a lost
acknowledgment retries the saved message with the same `messageId`. An acknowledged
reply resumes with `tasks/resubscribe` and its opaque SSE cursor; the displayed
partial answer is saved with that cursor. Answers to questions carry the pending
task ID. Switching conversations and unmounting cancel active readers.

The speech button obtains a participant token from `/api/token`. For a first call,
the backend creates the thread with the selected agent and endpoint, then passes
`thread_id` in the room dispatch metadata. Titles use the first message, shortened
locally to avoid another model request.

## Admin inspector

The explicit **Admin inspector** control opens lanes, tasks, journal, inbox, SQL,
and reply replay. Ordinary chat does not fetch database snapshots. The developer
backend calls the existing `livekit.AgentDBService/DumpDatabase` RPC at
`CONSOLE_AGENTDB_ADMIN_URL` using an `agent.databaseAdmin` credential, downloads its
consistent SQLite export, and validates storage rows against SDK-generated models.
The RPC path is `/twirp/livekit.AgentDBService/DumpDatabase`; configure the origin
of the existing service or its forwarding proxy. There is no separate database
management service in the console.

The SQL editor queries that exported snapshot on the backend. SQLite enforces
read-only access; queries have a two-second execution budget, a 10,000-row limit,
and exports have a 64 MiB limit. Snapshot inspection shows at most 1,000 rows per
table. These are local inspector limits, not framework transport budgets. No
project token, database-admin token, or signed export URL is returned to the browser.
Replay uses the agent's authorized A2A `tasks/resubscribe` endpoint rather than
assuming a task ID is a storage stream ID. The panel retains the latest 500 frames.
Delegation cards require explicit A2A `referenceTaskIds`; tool names, private extras,
call-ID suffixes, and prompt text are never used to infer a delegation.

## Checks and generated records

```sh
pnpm --dir console check
pnpm --dir console build
pnpm --dir console test:browser  # hermetic browser contract tests
uv run pytest console/test_console_api.py
```

The existing `test:e2e` suite requires the live example services and provider keys.
The hermetic browser suite uses the production build and mocked service responses.

JSON schemas and TypeScript records under `src/lib/generated` are derived from
installed SDK Pydantic models and the A2A SDK's generated protobuf descriptors.
Validate untrusted responses once through `src/lib/decode.ts`. To regenerate/check:

```sh
pnpm --dir console schema:generate
pnpm --dir console schema:check
uv run python -m grpc_tools.protoc -I ../protocol/protobufs \
  --python_out=console/generated ../protocol/protobufs/livekit_agentdb.proto
```

The last command regenerates the Python admin RPC records from LiveKit protocol.
Use the environment's `grpc_tools.protoc` so its output matches the protobuf runtime.
