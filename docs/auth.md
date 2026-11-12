# Authentication

Who may reach an agent's endpoints, its threads and their databases, and with which
credential. Every credential is a LiveKit access token: a JWT signed by a project API key,
carrying grants. A field in an A2A message, an agent card or its extensions never grants
access.

## Principals

| Principal                 | Holds                             | Reaches                                                                                     |
| ------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------- |
| the application's backend | a project API key and secret      | mints thread grants for its users; the threads service; dispatches calls with a `thread_id` |
| a browser or a device     | a thread grant its backend minted | the endpoints and the threads service, for the threads its grant covers                     |
| a deployed worker         | its agent API key and secret      | registers; mints its own job, run and delegation tokens                                     |
| an `lk agent dev` worker  | a worker token, no key            | registers; asks Cloud for every token it needs (`CreateJobToken`)                           |
| another agent, delegating | a delegation token                | the endpoint it calls, on the caller's thread                                               |
| an admin tool             | a token per call                  | the databases its token names; creating and listing them                                    |
| `lk agent console`        | nothing                           | a local listener on `127.0.0.1`, unauthenticated                                            |

## Grants

| Grant         | Claim                                                                              | Allows                                                                                       |
| ------------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| endpoint call | `agentEndpoint: {call, agentName?, deployment?}`                                   | reach an endpoint through a LiveKit node: its card and its HTTP routes; no thread data       |
| thread        | `agentThread: {scope \| threadIds \| all, list, create, read, write, delete, sql}` | the threads it covers, at the endpoints and the threads service; with `sql`, their databases |
| database      | `agentDatabase: {databaseIds, read, write, delete, create, list}`                  | the standalone databases it lists; with `create` and `list`, the project's                   |

`agentEndpoint` is reach, `agentThread` and `agentDatabase` are data. A thread grant
without `agentEndpoint` is refused at the node, so `Credentials.thread_grant_token()`
adds both. `Credentials.project_token()` carries `agentEndpoint` alone.

### The thread grant

```json
{
  "agentThread": {
    "scope": "user:42",
    "create": true,
    "write": true,
    "read": true
  }
}
```

| Field       | Meaning                                                                |
| ----------- | ---------------------------------------------------------------------- |
| `scope`     | every thread of one subject; the only form that can `create` or `list` |
| `threadIds` | the listed threads only                                                |
| `all`       | every thread of the project; enables no flag by itself                 |
| `list`      | the threads service's listing of the scope                             |
| `create`    | a first message with no `contextId`: a new thread in the scope         |
| `write`     | a message, `tasks/cancel`                                              |
| `read`      | `tasks/get`, `tasks/list`, `GetThreadItems`, `GetThreadTasks`          |
| `delete`    | deleting a thread                                                      |
| `sql`       | the thread's database on the data plane; never in a browser's grant    |

A grant with no subject (`scope`, `threadIds` or `all`) is refused. A thread belongs to the
project, not to an agent or a deployment: a thread grant names neither and works on every
door of the project.

### Threads and databases

A thread's database has the thread's id: `AT_...` names both, and nobody handles a
second id. A `DB_...` id names a standalone database, created with `CreateDatabase`, such
as a memory shared across threads; `agentDatabase.databaseIds` lists those.

| Database                   | Reached with                                       |
| -------------------------- | -------------------------------------------------- |
| a thread's, `AT_...`       | `agentThread` covering the thread, with `sql`      |
| a standalone one, `DB_...` | `agentDatabase` listing it, with `read` or `write` |

| Database grant | Allows                                                                     |
| -------------- | -------------------------------------------------------------------------- |
| `databaseIds`  | the standalone databases `read`, `write` and `delete` apply to             |
| `read`         | queries that write nothing, and their dumps; a write is refused            |
| `write`        | every query                                                                |
| `delete`       | deleting a listed database; a thread's goes only with its thread           |
| `create`       | creating a database; it names none, since agent-db picks the id            |
| `list`         | listing the project's databases                                            |

No grant reaches every database. An admin tool mints a token for the one it touches, as
for rooms.

The data plane takes the token on the WebSocket upgrade, as `access_token`, so a proxy
refuses the connection before its first frame.

## Where a worker's tokens come from

A worker never uses its key's full reach for a run. Every token it sends is scoped to the
thread the run belongs to:

| Token          | Grant                                                                                                          | Minted by a deployed worker | Minted for an `lk agent dev` worker |
| -------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------- | ----------------------------------- |
| a call's       | `agentThread {threadIds: [job.thread_id], read, write, sql}`                                                   | the worker, with its key    | Cloud, through `CreateJobToken`     |
| a text run's   | `agentThread {threadIds: [contextId], read, write, sql}`                                                       | the worker, with its key    | Cloud, through `CreateJobToken`     |
| a delegation's | `agentEndpoint.call`, `agentThread {threadIds: [caller's thread], read, write}`, identity `delegate`, one hour | the worker, with its key    | Cloud, through `CreateJobToken`     |

Cloud mints a dev worker's tokens for the job's thread only; the worker cannot name
another. A deployed worker's key is unscoped, so on a deployed worker the per-thread bound
is the framework's discipline, not a boundary.

## How a thread id travels

The thread is named by whoever already holds authority over it, and every hop carries a
token that covers it. The agent never picks a thread.

### Text

```
browser ──(thread grant, contextId)──▶ LiveKit node ──(verified principal)──▶ worker ──(thread token, sql)──▶ agent-db
```

1. The application's backend mints a thread grant for its user: `scope=<user>` to let the
   browser start and list threads, `threadIds=[...]` to pin it to some.
2. The browser sends `message/send` with `Authorization: Bearer <grant>` and the thread
   as `contextId`; with no `contextId`, `create` makes a new thread in the grant's scope.
3. The node verifies the signature against the project's keys and the `agentEndpoint`
   grant, and forwards the request with the token as sent.
4. The worker reads the token's `agentThread` and checks it covers `contextId` with the
   flag the method needs; a thread outside it is `403`.
5. The run opens that thread's database with a token for that thread.

### Calls

A call's thread is its job's `thread_id`, set by the dispatch
([threads.md](threads.md#calls)). Every dispatch route is signed by a project key, so the
id is trusted without a grant of its own. A job without `thread_id` gets no thread token
and runs in memory.

### Delegation

A `RemoteAgent` tool shares its caller's thread with the endpoint it calls, under a
delegation token from the worker's token source (above). The endpoint answers in its own
partition of that thread. A caller with no thread sends `agentEndpoint` alone, enough to
read the card. An endpoint off the project gets no LiveKit token: the application passes
the credential it expects as `token=`, and the first message carries text only.

## Endpoints

| Request                                    | `agentEndpoint` alone | Thread grant                       |
| ------------------------------------------ | --------------------- | ---------------------------------- |
| `GET .well-known/agent-card.json`          | yes                   | yes                                |
| `message/send`, `message/stream`           | `403`                 | `write`; `create` for a new thread |
| `tasks/get`, `tasks/list`                  | `403`                 | `read`                             |
| `tasks/cancel`                             | `403`                 | `write`                            |
| an `@server.http` route                    | yes                   | yes                                |
| an `@server.http` route with `public=True` | no token needed       | no token needed                    |

A refusal is an HTTP status with `{"error": "..."}`, outside JSON-RPC: `401` for no token
or one no project key signed, `403` for a token whose grants do not cover the call.

### Verification

A project has several API keys; only the server knows them all.

| Worker reached                                               | Signature verified by                                                                           | The worker                                                                                                                            |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| through a LiveKit node (Cloud or self-hosted livekit-server) | the node, against every key of the project; a token it cannot verify is `401` before the worker | reads the claims of the token the node verified and authorizes: thread, method, flag                                                  |
| directly, on a self-hosted or OSS deployment                 | nobody                                                                                          | reads the token's claims unverified and authorizes the same way; it logs a warning at startup that its endpoints are in insecure mode |
| `lk agent console`                                           | nobody                                                                                          | every caller holds every thread permission in the `console` scope                                                                     |

The worker never verifies a signature. It reads the token from `Authorization: Bearer`,
else the `access_token` query parameter: the one the node verified, and no other.

## What stays on the application's backend

- Its API key and secret.
- Choosing the thread: which threads a user may reach, stored against the user in the
  application's own tables, and named in the grants and dispatches it mints.
- `agentDatabase`.

The browser receives only a thread grant without `sql`. It never sees a key, never sets
attributes, and never reads another scope.
