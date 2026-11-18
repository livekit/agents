// The page's routes sit next to it: /agents/<agent>/<deployment>/{token, chat/v1}.
const base = location.pathname.replace(/\/?$/, "/");
const browser = (localStorage.browser ??= crypto.randomUUID());
const $ = (id) => document.getElementById(id);
let grant = null;
let thread = null;

// A thread grant for this browser's scope, renewed once it expires.
async function auth() {
  if (!grant || Date.now() > grant.expires) {
    const res = await fetch(`${base}token?browser=${browser}`);
    if (!res.ok) throw new Error(`token: ${res.status}`);
    grant = await res.json();
    grant.expires = Date.now() + (grant.ttl - 30) * 1000;
  }
  return grant;
}

// The threads REST of LiveKit's public API; `p_` is the grant's own project.
async function threadsApi(path) {
  const { api, token } = await auth();
  const res = await fetch(`${api}/v0/projects/p_/threads${path}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error(`threads API: ${res.status}`);
  return res.json();
}

async function listThreads() {
  const { scope } = await auth();
  const { items = [] } = await threadsApi(`?scope=${encodeURIComponent(scope)}&page.pageSize=50`);
  $("threads").replaceChildren(...items.map((t) => {
    const li = document.createElement("li");
    li.textContent = new Date(t.createdAt).toLocaleString([], {
      dateStyle: "medium",
      timeStyle: "short",
    });
    li.classList.toggle("open", t.threadId === thread);
    li.onclick = () => open(t.threadId).catch(showError);
    return li;
  }));
}

// Renders a thread from its items, the source of truth: user and assistant messages only.
async function open(id) {
  thread = id;
  const items = [];
  let cursor = "";
  do {
    const page = await threadsApi(`/${id}/items?page.pageSize=100&page.cursor=${cursor}`);
    items.push(...(page.items ?? []));
    cursor = page.pageInfo?.nextCursor ?? "";
  } while (cursor);
  $("messages").replaceChildren();
  for (const { item } of items) {
    const message = item?.message;
    if (message?.role === "USER" || message?.role === "ASSISTANT") {
      bubble(message.role.toLowerCase(), message.content.map((c) => c.text ?? "").join(""));
    }
  }
  await listThreads();
}

function bubble(role, text) {
  const el = document.createElement("div");
  el.className = `bubble ${role}`;
  el.textContent = text;
  $("messages").querySelector(".hint")?.remove();
  $("messages").append(el);
  el.scrollIntoView({ block: "end" });
  return el;
}

function showError(err) {
  bubble("assistant pending", err.message);
}

// The JSON of each `data:` line of a server-sent event stream.
async function* events(res) {
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    const lines = (buffer += value).split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (line.startsWith("data:")) yield JSON.parse(line.slice(5));
    }
  }
}

// A2A SendStreamingMessage to the chat endpoint: the answer streams into its bubble, then
// the thread is rendered again. With no contextId the platform starts a new thread.
async function send(text) {
  bubble("user", text);
  const reply = bubble("assistant pending", "…");
  const { token } = await auth();
  const res = await fetch(`${base}chat/v1`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "A2A-Version": "1.0",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: 1,
      method: "SendStreamingMessage",
      params: {
        message: {
          messageId: crypto.randomUUID(),
          role: "ROLE_USER",
          parts: [{ text }],
          ...(thread && { contextId: thread }),
        },
      },
    }),
  });
  if (!res.ok) throw new Error((await res.json()).error ?? `chat: ${res.status}`);

  let answer = "";
  let contextId = thread;
  for await (const { result, error } of events(res)) {
    if (error) throw new Error(error.message);
    const event = result.task ?? result.artifactUpdate ?? result.statusUpdate;
    contextId = event.contextId;
    for (const part of result.artifactUpdate?.artifact.parts ?? []) {
      reply.textContent = answer += part.text ?? "";
      reply.classList.remove("pending");
    }
    // a question or a failure comes as the final status's message
    const parts = result.statusUpdate?.status.message?.parts;
    if (parts && !answer) reply.textContent = parts.map((p) => p.text ?? "").join("");
  }
  await open(contextId);
}

$("composer").onsubmit = async (event) => {
  event.preventDefault();
  const text = $("text").value.trim();
  if (!text) return;
  $("text").value = "";
  event.submitter.disabled = true;
  try {
    await send(text);
  } catch (err) {
    showError(err);
  } finally {
    event.submitter.disabled = false;
    $("text").focus();
  }
};

$("new").onclick = () => {
  thread = null;
  const hint = document.createElement("p");
  hint.className = "hint";
  hint.textContent = "Ask anything.";
  $("messages").replaceChildren(hint);
  listThreads().catch(showError);
  $("text").focus();
};

listThreads().catch(showError);
