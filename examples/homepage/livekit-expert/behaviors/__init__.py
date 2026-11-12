"""The agent's session behaviors — when X happens, the agent does Y.

Each behavior is one module in this directory, named after what it makes the
agent do. Session event behaviors expose an attach function ``(session) -> None``;
immediate actions use plain verb names. Any authored language lives in the
central ``prompts`` package. The job is ambient — a behavior that needs the room
reaches it through the SDK's ``get_job_context()``.

The entrypoint attaches behaviors by calling them, one line each, where the
session is created — those lines are the agent's behavioral table of
contents. Adding a behavior is adding a module here and one line there;
removing one is the reverse.
"""
