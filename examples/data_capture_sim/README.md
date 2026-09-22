# Data Capture Simulation

A simulation harness for the beta data-capture workflows (`livekit.agents.beta.workflows`).
The agent collects one detail at a time — email, phone number, mailing address, date of
birth, name, credit card — by handing each off to the matching capture task, with
confirmation required so every confirm and correction branch runs even in text-only
simulations.

`scenarios.yaml` drives the simulated users. The `data-capture-sim` workflow runs them on
every change to the workflows package or this directory.

## Running

```bash
uv run lk agent simulate examples/data_capture_sim/src/agent.py \
  --scenarios examples/data_capture_sim/scenarios.yaml
```

For setup instructions, see the [main examples README](../README.md).
