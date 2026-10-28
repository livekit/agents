/** The fields used by the console's backend responses and inspector views. */
import type { Message, Artifact } from "./a2a";

export type StreamAvailability = { url: string; available: boolean; error?: string | null };
export type ConsoleConfig = { grant: { subs: string[] }; streams: StreamAvailability };
export type Grant = {
  token: string;
  agent: string;
  sub: string;
  endpoint: string;
  endpointName: string;
  expiresAt: number;
};
export type SessionRow = {
  sessionId: string;
  agentName: string;
  sub: string;
  attributes: Record<string, string>;
  region?: string;
  databaseId: string;
  createdAt: number;
  lastActive: number;
  expiresAt: number;
  idleTtlSeconds: number;
};
export type ChatItem =
  | {
      type: "message";
      role: "developer" | "system" | "user" | "assistant";
      content: unknown[];
      created_at?: number;
      id?: string;
    }
  | {
      type: "function_call";
      call_id: string;
      name: string;
      arguments: string;
      created_at?: number;
      id?: string;
    }
  | {
      type: "function_call_output";
      call_id: string;
      output: string;
      is_error: boolean;
      created_at?: number;
      id?: string;
    }
  | { type: "agent_handoff"; new_agent_id: string; created_at?: number; id?: string }
  | { type: "agent_config_update"; created_at?: number; id?: string };
export type Lane = {
  lane: string;
  version: number;
  state?: unknown;
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
  payload: ChatItem;
};
export type JournalEvent = {
  id: number;
  ts_ms: number;
  lane: string;
  run_id: string;
  type: string;
  payload: unknown;
};
export type InboxRow = {
  id: number;
  lane: string;
  task_id: string;
  message_id: string;
  message: Record<string, unknown>;
  received_at: number;
};
export type TaskRow = {
  task_id: string;
  lane: string;
  seq: number;
  name: string | null;
  message_id: string | null;
  status:
    "submitted" | "working" | "input-required" | "completed" | "failed" | "canceled" | "rejected";
  updated_at: number;
  data: {
    history?: Message[];
    artifacts?: Artifact[] | null;
    metadata?: Record<string, unknown>;
    status_message?: Message | null;
  };
  ask: {
    question: string;
    tool?: string | null;
    call_id?: string | null;
    answer_tool?: string | null;
  } | null;
};
export type DatabaseDump = {
  databaseId: string;
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
  database?: DatabaseDump | null;
  databaseError?: string | null;
  streams: StreamAvailability;
};
export type History = { items: ChatItem[] };
export type Sessions = { sessions: SessionRow[]; before?: string };
export type CreatedSession = { session: SessionRow };
export type QueryResult = {
  columns: string[];
  rows: unknown[][];
  truncated: boolean;
  tip: number;
  ms: number;
};
export type Example = {
  name: string;
  agent: string;
  endpoint: string;
  description: string;
  voice: boolean;
  start: string;
  dir: string;
  url: string;
  live: boolean;
};
export type Examples = { examples: Example[] };
export type CallRequest = {
  room_name?: string | null;
  participant_name?: string | null;
  participant_identity?: string | null;
  participant_attributes?: Record<string, string> | null;
  room_config: { agents: { agent_name: string; metadata: string }[] };
};
export type CallMetadata = { session_id?: string | null; sub?: string | null; endpoint: string };
export type CallToken = { server_url: string; participant_token: string; session_id: string };
export type TitleRequest = { sessionId: string; firstMessage: string };
export type Json = unknown;

/** One line of an expert's progress, and when it arrived (ms). */
export type Progress = { at: number; text: string };

/** A view of an agent task explicitly referenced by an A2A message. */
export type Delegation = {
  /** The agent's name. */
  name: string;
  /** The request as sent to the expert. */
  task: string;
  taskId?: string;
  progress: Progress[];
  answer?: string;
  /** The call's own output was an error. */
  failed?: boolean;
  /** When the call's output was recorded (ms). */
  endedAt?: number;
};

/** What every entry carries: its id, and when it happened (ms), so voice turns interleave with text. */
type Stamped = { id: string; at: number };

/** Milliseconds from the send: the acknowledgment, the first piece of the reply, the end. */
export type Timing = { ack?: number; first?: number; done?: number };

/** The conversation as the stage shows it: one entry per thing that happened. */
export type Entry =
  | (Stamped & { kind: "user"; text: string; voice?: boolean })
  | (Stamped & {
      kind: "agent";
      text: string;
      taskId?: string;
      streaming?: boolean;
      voice?: boolean;
      interrupted?: boolean;
      timing?: Timing;
    })
  | (Stamped & { kind: "progress"; text: string })
  | (Stamped & { kind: "question"; text: string })
  | (Stamped & { kind: "delegation" } & Delegation)
  | (Stamped & {
      kind: "status";
      text: string;
      taskId?: string;
      state?: string;
      failed?: boolean;
    });
