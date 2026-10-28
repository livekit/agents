/** Validated SDK records shared by browser views. */
export type { ConsoleConfig } from "./generated/ConsoleConfig";
export type { Grant } from "./generated/Grant";
export type { SessionRow } from "./generated/SessionRow";
export type {
  SessionDump, DatabaseDump, LaneRow as Lane, Item, EventRow as JournalEvent,
  InboxRow, TaskRow,
} from "./generated/SessionDump";
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
