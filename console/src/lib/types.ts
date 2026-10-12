/**
 * The shapes the console API (`examples/console_api.py`) answers with, in the plan's
 * words (thread-registry-plan.md 2 and 5.1): a session, its lanes, the
 * items, the journal, the inbox and the tasks. The API reads the documented
 * schema and promises nothing about its stability, so everything optional
 * here is read defensively.
 */

export type Json = unknown;

export type ConsoleConfig = {
  registryUrl: string;
  mgmtUrl: string;
  wsUrl: string;
  streamsUrl: string;
  streams: { url: string; available: boolean; error?: string };
  grant: { subs: string[] };
};

export type Grant = {
  token: string;
  agent: string;
  sub: string;
  /** The endpoint's URL, for the A2A client. */
  endpoint: string;
  /** The endpoint's name on the agent, as threads are tagged with it. */
  endpointName: string;
  expiresAt: number;
};

export type SessionRow = {
  sessionId: string;
  agentName: string;
  sub?: string;
  attributes?: Record<string, string>;
  region: string;
  databaseId?: string;
  createdAt: number;
  lastActive: number;
  expiresAt: number;
  idleTtlSeconds: number;
};

export type Lane = {
  lane: string;
  version: number;
  state: Json;
  agent_stack: Json;
  lease_holder: string | null;
  lease_until: number;
  waiter: string | null;
  drained_id: number;
};

export type Item = {
  lane: string;
  id: string;
  seq: number;
  type: string;
  message_id: string | null;
  payload: Json;
};

export type JournalEvent = {
  id: number;
  ts_ms: number;
  lane: string;
  run_id: string;
  author: string;
  type: string;
  payload: Json;
};

export type InboxRow = {
  id: number;
  lane: string;
  task_id: string;
  message_id: string;
  message: Json;
  received_at: number;
};

export type TaskStatus =
  | "submitted"
  | "working"
  | "input-required"
  | "completed"
  | "failed"
  | "canceled"
  | "rejected";

export type TaskRow = {
  task_id: string;
  lane: string;
  seq: number;
  /** The AgentTask class, for a background task; null for a message's task. */
  name: string | null;
  message_id: string | null;
  status: TaskStatus | string;
  status_message: Json;
  updated_at: number;
  /** The A2A task's history, artifacts and metadata. */
  data: { history?: Json; artifacts?: Json; metadata?: Json } | null;
  internal: Json;
};

export type DatabaseDump = {
  databaseId: string;
  region?: string;
  createdAt?: number;
  expiresAt?: number;
  tip: number;
  tables: string[];
  schemaVersion: number;
  lanes: Lane[];
  items: Item[];
  events: JournalEvent[];
  inbox: InboxRow[];
  tasks: TaskRow[];
};

export type SessionDump = {
  fetchedAt: number;
  session: SessionRow;
  database?: DatabaseDump;
  databaseError?: string;
  streams: { url: string; available: boolean; error?: string };
};

/** One line of an expert's progress, and when it arrived (ms). */
export type Progress = { at: number; text: string };

/**
 * A delegation as the thread shows it: the call that handed a request to an
 * expert over A2A, what the console learnt from the wire while it ran, and
 * what the call's history said about it afterwards (`lk.delegation.*`).
 * The expert's own task row, from the session's database, completes it.
 */
export type Delegation = {
  /** The tool that delegated: `delegate` on a call, `coverage` by text. */
  name: string;
  /** The request as sent to the expert. */
  task: string;
  /** The delegating call's id, once the history says which. */
  callId?: string;
  /** The expert's task id, once known. */
  taskId?: string;
  endpoint?: string;
  transport?: string;
  profile?: string;
  progress: Progress[];
  answer?: string;
  /** The call's own output was an error. */
  failed?: boolean;
  /** When the call's output was recorded (ms). */
  endedAt?: number;
};

/** What every entry carries: its id, and when it happened (ms), so voice turns interleave with text. */
type Stamped = { id: string; at: number };

/** The conversation as the stage shows it: one entry per thing that happened. */
export type Entry =
  | (Stamped & { kind: "user"; text: string; voice?: boolean })
  | (Stamped & { kind: "agent"; text: string; taskId?: string; streaming?: boolean; voice?: boolean; interrupted?: boolean })
  | (Stamped & { kind: "tool"; name: string; args?: string; output?: string; isError?: boolean })
  | (Stamped & { kind: "progress"; text: string })
  | (Stamped & { kind: "question"; text: string })
  | (Stamped & { kind: "delegation" } & Delegation)
  | (Stamped & { kind: "status"; text: string; taskId?: string; state?: string; meta?: string; failed?: boolean });
