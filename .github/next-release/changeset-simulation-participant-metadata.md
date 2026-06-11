---
"livekit-agents": patch
---

`JobContext.simulation_context()` now resolves the `SimulationDispatch` from the simulator participant's `lk.simulator.dispatch` attribute instead of the agent's job dispatch metadata, leaving `job.metadata` and `room.metadata` free for user payloads under simulation — an agent that reads its own dispatch metadata now behaves identically in production and under simulation. The job/room metadata path is kept as a fallback for older servers and for `fake_job_context` in tests.
