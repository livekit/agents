const browser = (localStorage.browser ??= crypto.randomUUID());
const $ = (id) => document.getElementById(id);
const grants = {};
let thread = null;

// A token to send, from backend.py and renewed once it expires: for one thread of this browser,
// or with no thread, to start a new one.
async function auth(id = "") {
  const grant = grants[id];
  if (grant && Date.now() < grant.expires) return grant;
  const fresh = await get(`token?browser=${browser}${id && `&thread=${id}`}`);
  fresh.expires = Date.now() + (fresh.ttl - 30) * 1000;
  return (grants[id] = fresh);
}

// The app's backend, backend.py: tokens, and the threads it reads for the page.
const BACKEND = "http://localhost:8001";

async function get(path) {
  const res = await fetch(`${BACKEND}/${path}`);
  if (!res.ok) throw new Error(`${path.split("?")[0]}: ${res.status}`);
  return res.json();
}

async function listThreads() {
  const threads = await get(`threads?browser=${browser}`);
  $("threads").replaceChildren(...threads.map((t) => {
    const li = document.createElement("li");
    li.textContent = new Date(t.created_at * 1000).toLocaleString([], {
      dateStyle: "medium",
      timeStyle: "short",
    });
    li.classList.toggle("open", t.id === thread);
    li.onclick = () => open(t.id).catch(showError);
    return li;
  }));
}

// Renders a thread from its messages, the source of truth.
async function open(id) {
  thread = id;
  const messages = await get(`threads/${id}?browser=${browser}`);
  $("messages").replaceChildren();
  for (const { role, text } of messages) bubble(role, text);
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
  const { token, agent } = await auth(thread ?? "");
  const res = await fetch(`${agent}/chat/v1`, {
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
  hint.textContent = "How can I help?";
  $("messages").replaceChildren(hint);
  listThreads().catch(showError);
  $("text").focus();
};

listThreads().catch(showError);
