import type { TaskRow } from "./types";

/** Explicit A2A task references are the only evidence that another task was involved. */
export function referencedTasks(tasks: TaskRow[], agent: string): TaskRow[] {
  const references = new Set(tasks.filter((task) => task.lane === agent)
    .flatMap((task) => task.data.history ?? [])
    .flatMap((message) => message.referenceTaskIds ?? []));
  return tasks.filter((task) => references.has(task.task_id) && task.lane !== agent);
}
