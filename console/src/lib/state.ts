/**
 * The A2A state of a task as a dot: bright while it is being worked,
 * attention while it waits on the person, failed for the three bad ends; a
 * completed task is plain. One place, so the Tasks table and the delegation
 * card agree.
 */
export const STATE_DOT: Record<string, string> = {
  submitted: "bg-ink",
  working: "bg-ink",
  "input-required": "bg-attention",
  completed: "bg-label",
  failed: "bg-failed-text",
  canceled: "bg-failed-text",
  rejected: "bg-failed-text",
};

export const stateDot = (status: string | undefined) => (status && STATE_DOT[status]) || "bg-control";
