# The backend API: planned

Not built. This page extends [backend-api.md](backend-api.md), which
describes the threads client, the tokens and `AgentClient` as they work
today.

## A thread handle

One handle over the platform's threads service and the agent's
endpoint, with the project credential or a thread grant:

```python
from livekit import api
lk = api.LiveKitAPI()

thread = lk.threads.get(thread_id)                                       # a handle; nothing fetched yet
thread = await lk.threads.create(scope=user_id)                          # a new thread for a subject
threads = await lk.threads.list(scope=user_id, limit=30)                 # newest first, with info loaded

thread.id
info = await thread.info()                                                 # scope, attributes, created, last active
await thread.update(attributes={"title": "Renewal"})
await thread.delete()

items = await thread.chat_history()                       # list[ChatItem]: the thread's history
items = await thread.chat_history(agent="eligibility")    # one agent's lane of the thread
tasks = await thread.background_tasks()                   # the thread's background tasks
```

`create`, `get`, `list`, `info`, `update`, `delete`, `chat_history` and
`background_tasks` go to the threads service (`GetThreadItems`,
`GetThreadTasks` for the last two); `send` and `submit` to the agent's
endpoint, which must be deployed. Inside a run, `ctx.thread` is the same
class with more: `lane()`.

## Calls

```python
dispatch = api.CreateAgentDispatchRequest(agent_name="sales", room=room, thread_id=thread.id)
await lk.agent_dispatch.create_dispatch(dispatch)                              # the job carries the thread
await lk.sip.create_sip_participant(api.CreateSIPParticipantRequest(room_name=room, sip_trunk_id=TRUNK, sip_call_to=lead.phone))
```

The dispatch carries the thread id, and the job hands it to the call's
`LiveSession`.

## Tasks

```python
tasks = await thread.background_tasks()
await thread.cancel(task_id)
await thread.submit(agent="sales", task=FollowUp(lead_id=lead.id), delay=timedelta(days=2))   # no model
```

Submitting from outside comes with background tasks
([planned-agent-tasks.md](planned-agent-tasks.md)).

## Webhooks

```python
await client.watch(task_id, webhook=url)   # a POST when the task changes state
```

`AgentClient.watch` registers a push notification config for a task:
`tasks/pushNotificationConfig/set` on the endpoint.

## From the browser

A chat UI needs no backend of its own beyond the one call that mints a
token ([backend-api.md](backend-api.md), Tokens). The browser uses the
same client with it:

```javascript
const threads = new ThreadsClient({ token });             // scoped: this subject
const chats = await threads.list({ limit: 30 });           // only this subject's
const thread = threads.get(chats[0].id);
const history = await thread.chatHistory();
for await (const event of thread.send("Where is my order?")) { ... }
```

With a grant, `list` takes the subject from the token (the grant's `list`
flag) and `delete` needs the grant's `delete` flag.

See `examples/use-cases/planned-09_speed_to_lead.py` and
`planned-12_many_chats.py`.
