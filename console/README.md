# Console

A local developer console for talking to agents by text or voice and inspecting
conversations. Project credentials stay on the backend; the browser receives
scoped thread grants. Agent requests use authenticated A2A 1.0 POST streams.

Run the example workers and LiveKit services first. Set the variables in
`console/.env.example` for both servers and share a random `CONSOLE_API_TOKEN`.
Next.js reads `console/.env.local`; Python reads its environment.

From the repository root:

```sh
uv run python console/console_api.py
pnpm --dir console dev
```

Open `http://localhost:3000`. Both servers bind to loopback. Privileged routes
accept only `CONSOLE_ORIGIN` (default `http://localhost:3000`). The example picker
reads `examples/*/console.toml`. Conversation lists and history use the threads
service. Reloading retries an unacknowledged message by its original ID or resumes
an acknowledged task by its saved cursor. Calls join the selected conversation.

The **Admin inspector** loads snapshots only when opened. Configure
`CONSOLE_AGENTDB_ADMIN_URL` for the existing AgentDB admin service. The backend
uses LiveKit SDK token minting and `livekit.protocol.agentdb` records for
`AgentDBService/DumpDatabase`; install a protocol package exposing `DumpRequest`
and `DumpResponse`. Admin tokens and signed export URLs stay on the backend.
SQLite enforces read-only SQL with a two-second budget, a 10,000-row result limit,
a 64 MiB export limit, and at most 1,000 rows per inspector table.

The console uses TypeScript view types. Browser requests, saved messages, and
external A2A replies receive small runtime checks where they enter the app.

```sh
pnpm --dir console check
pnpm --dir console build
pnpm --dir console test:browser
uv run pytest console/test_console_api.py
```

`check` runs TypeScript, lint, the theme guard, and unit tests. The browser suite
uses the production build with mocked services in Chromium and WebKit. Python
tests cover authorization, snapshot credentials, and read-only SQL. `test:e2e`
requires live example services and provider keys.
