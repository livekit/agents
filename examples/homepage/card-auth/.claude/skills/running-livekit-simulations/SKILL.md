---
name: running-livekit-simulations
description: 'Runs LiveKit agent simulations and acts on the results. Use when the user says "run my simulations", "regression test my agent before deploying", "run the scenarios", "use lk agent simulate", "did my agent pass", "why did this scenario fail", "run simulations in CI", "test the audio pipeline", "check turn-taking and interruptions", or wants to check whole-conversation behavior before shipping. Covers text and audio mode and what each catches, running against a local or deployed agent, degraded-audio flags, automating a pre-release run, and triaging failures with list, view and export. For writing the scenarios use writing-livekit-scenarios. Not the default for a bare "test my agent", which goes to debugging-livekit-agents. Use this skill when the user names simulations, scenarios, a run, CI, or shipping.'
license: MIT
metadata:
  author: livekit
---

# Running LiveKit simulations

A simulation plays a scenario against the real agent using an LLM-driven simulated user, then a
judge grades the transcript. A unit test asserts on one turn. A simulation tells you whether a
*whole conversation* reached the right outcome.

Read `lk agent simulate --help` before running. Subcommands and flags change, a wrong flag wastes a
paid run, and this skill doesn't restate them. `reading-livekit-docs` has the rest.

## When to reach for a simulation

Use simulations to regression-test long-horizon behavior before deploying to production: whether a
multi-turn conversation reaches the right outcome when the caller backtracks, whether details
gathered early survive to the end, whether the agent holds to its instructions under pressure, and
whether it ended in the right *state*. For a single turn, use `testing-livekit-agents`; to poke at
behavior while editing, use `debugging-livekit-agents`.

## Running

Run from the agent's project directory. The mode is a subcommand:

```bash
lk agent simulate text --scenarios scenarios.yaml    # see --help for the current flags
```

With no scenario file, the CLI generates scenarios from the agent's source. That **uploads the
code**, and the CLI asks for confirmation first. Generation belongs to `writing-livekit-scenarios`.

By default the CLI starts the agent as a local worker, dispatches the scenarios to it, and stops it
when the run ends. An option lets you grade an already-running agent by name instead. That needs a
scenario file, since there's no local source to generate from.

Concurrency is limited per run and per project. The docs have the current limits.

## Text or audio

**Text is the default**, and it's the right one. The simulated user exchanges text with the agent,
so the run exercises the LLM, the tools and the conversation logic while the framework turns off
STT, TTS and VAD. It's faster, cheaper and more deterministic. Use it for iteration and for anything
automated.

**Audio runs the same scenarios through the full speech pipeline.** The simulated user speaks,
listens and interrupts like a caller would, and the run scores what only speech exposes:

- **Turn-taking**: starting to speak before the caller has finished, or leaving a caller who has
  finished waiting.
- **Interruption handling**: yielding to a barge-in, and telling a brief acknowledgment apart from a
  new turn.
- **Transcription accuracy** in both directions, scored separately for the things that matter:
  names, numbers, addresses, confirmation codes.
- **Perceived latency**: what the caller heard, which differs from what the agent reports about
  itself. The gap between the two is what the user experiences.

Audio runs execute in real time, call the STT and TTS providers every turn, and are metered at a
higher rate. Save them for a release candidate or a change that touches speech, turn-taking or
interruption. Don't put them in a recurring job.

The audio subcommand has options to degrade the simulated caller's audio (noise, a poor microphone,
packet loss). Use them to test what the agent does with speech it can't hear clearly. It should ask
for a repeat instead of guessing. Combine them for a worst-case caller.

## Automating a pre-release run

All you need is a committed scenario file and a scheduled or release-branch job. The CLI prints
plain output when it isn't attached to a terminal and exits non-zero when any scenario fails, so the
job fails without extra wiring; the docs have a worked CI example to start from. Keep automated runs
in text mode. Every scenario in the committed file has to pass or the job fails, so keep aspirational
scenarios the agent doesn't pass yet in a separate file you run on demand.

## Reading the results

A run prints a verdict per scenario and a dashboard link. The verdict tells you what happened; the
transcript tells you why, so work from the transcript. The dashboard link is for the human. Your
path is `export`: it prints a finished run, with each scenario's full chat context, as JSON — read a
failing transcript from there, diff two runs, or archive a run as a build artifact. `list` finds the
run id and also has machine-readable output; `--help` names the flags.

**To triage a failure, decide which of these it is:**

1. **A bug in the agent.** Fix the agent and re-run. Check instructions and tool descriptions first.
   A failure that looks like bad reasoning is often a tool whose description never says when to
   use it.
2. **A bad scenario.** The expectation requires something the agent shouldn't do, or it's too vague
   for a judge to decide consistently. Fix the scenario. A vague `agent_expectations` is the most
   common reason a verdict flips between runs.
3. **A gap in the simulated world.** The scenario passes but production failed, or the other way
   round. Usually the agent reached different data, the derived instructions leave out the turn
   that caused the problem, or the failure only happens in audio and a text run can't see it.

**After a fix, run the whole file**, not only the scenario you were working on. A fix for one
conversation often changes a neighbouring one.

**Move repeat failures down the stack.** A scenario that fails the same way every time is
describing a turn-level bug. A unit test pins it more cheaply and catches it earlier. See
`testing-livekit-agents`.

## Related skills

- Authoring and organizing scenarios: `writing-livekit-scenarios`
- Interactive debugging of a failure: `debugging-livekit-agents`
- Cheaper per-commit coverage: `testing-livekit-agents`
- Deploying the agent the run is grading: `operating-livekit-agents`
- Flags, versions, changelogs: `reading-livekit-docs`
