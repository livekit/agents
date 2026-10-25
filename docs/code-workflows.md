# Code workflows: deferred

The current proposal covers [conversational agent tasks](agent-tasks.md) and
[persisted application context](state-and-deps.md). Graphs, code workflows,
and a durable execution runtime are outside this phase.

A future code workflow could choose deterministic steps, branches, and loops
without having an LLM choose every transition. Ordinary Python awaits do not
provide durable resume points. Before adding an API, validate a real use case
and compare using an existing workflow engine.

Workflow authoring, scheduling, and static workflow diagrams remain future work.
