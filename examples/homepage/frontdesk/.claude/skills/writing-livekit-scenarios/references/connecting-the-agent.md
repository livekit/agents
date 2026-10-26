# Connecting scenarios to the agent

## Contents
- Detect the simulation, change nothing else
- Seed state from per-scenario data
- Mock tools for the session's lifetime
- Grade the final state, not just the transcript
- Which scenarios need this
- Check it took effect

Two problems make simulation suites hard to trust. Both are fixed in the agent's code, not the
scenario file:

1. **Nondeterminism.** A scenario that reaches the real backend grades differently every run and
   goes stale as that backend's data changes.
2. **A judge that only reads the conversation.** A smooth, confident conversation can still have
   booked the wrong room, and if the transcript is the only verdict, that run passes.

Look up the current API names with `reading-livekit-docs` before writing any of this. The docs'
section on connecting scenarios to your agent has working code in each language. This reference
describes the shape, which is stable. The names change, so it doesn't restate them.

## Detect the simulation, change nothing else

The agent asks the job context whether it's running under a simulation. Under one, it gets a
simulation context carrying the scenario and its per-scenario data. In production it gets nothing
back. That one branch is the whole integration:

- **Under simulation**, build state from the scenario's data and install mocks.
- **Otherwise**, connect to the real backends as before.

Put the branch at the top of the entrypoint, right after connecting. The context is available from
dispatch, so it's ready as soon as the entrypoint runs.

**Leave the production path untouched.** If wiring up simulations changes how the agent behaves in a
real session, the simulation is no longer testing what you ship.

## Seed state from per-scenario data

Each scenario can carry an arbitrary nested mapping (the docs call it userdata) that's passed to
the agent at runtime. That's how one scenario says "these rooms are free that day" and the next says
"nothing is free".

- **Keys arrive exactly as written** in the scenario file. Read them under the same names. There's
  no case conversion.
- **Seed the backend instead of special-casing the agent.** Build a fake calendar, an in-memory
  database, or a fixture inventory that the agent's real tools read through unchanged. Once the
  agent's *logic* branches on simulation state, the test is no longer testing the agent.
- **Pin the clock.** Scenarios written against absolute dates need the agent's idea of "now" fixed,
  usually from an environment variable, optionally overridden per scenario. Otherwise the suite
  starts failing as the calendar moves and nothing tells you why.

## Mock tools for the session's lifetime

A simulation runs a whole session by itself. Nothing wraps the turns the way a test does, so mocks
have to be installed for the session's lifetime instead of scoped to a block. The tool-mocking
helper has a form for this case.

The model still sees the real tool schemas and only execution is intercepted, so tool *selection*
stays under test. That's usually what you most want graded.

If a language has no session-scoped mocking helper, get the same result by defining tools inside
the entrypoint so they close over the state you seeded. The principle is the same: real schemas,
seeded data.

## Grade the final state, not just the transcript

Register the end-of-simulation callback and compare the agent's final state against what the
scenario said should happen, which by convention is an expected-state block in the scenario's data.
If they differ, fail the run with a reason.

The semantics tend to surprise people:

- **Your check can only fail a run.** The result is the AND of the judge's verdict and yours. You
  can fail a run the judge passed. You can't rescue one the judge failed.
- **Doing nothing is valid.** If you don't fail the run, the judge's verdict stands. Scenarios with
  no expected state should return early and be graded on the conversation alone.
- **The judge's verdict is readable** inside the callback, if you want to log it or make your check
  depend on it.

Compare at the level you care about: room *type*, booking *status*, followup *kind*. Asserting on
every field of a record gives you failures about formatting instead of behavior.

With this in place, a run passes only if the conversation was right *and* the agent ended in the
right state.

## Which scenarios need this

Not all of them, which is a good reason to keep separate scenario files:

- **Tool-flow scenarios** that drive a concrete outcome need seeded state and final-state grading.
  They're also the ones to write in point-form direction so the path stays stable.
- **Open-ended and adversarial scenarios** are graded on the conversation alone. Seeding adds
  nothing, and most need no per-scenario data at all.

## Check it took effect

When this fails silently it's expensive. The run looks fine while the agent talks to production
data, or a key is misspelled and the fake backend comes up empty.

Before trusting a suite, confirm once that a seeded scenario used the seeded state. Run it with
`debugging-livekit-agents`, or read a run's transcript and check that the agent offered what you
seeded and not what the real backend holds. Then confirm that an expected state you know is wrong
fails the run. If the state check can't fail, it isn't checking anything.
