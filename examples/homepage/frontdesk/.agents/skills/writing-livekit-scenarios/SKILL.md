---
name: writing-livekit-scenarios
description: 'Creates and maintains the scenarios a LiveKit agent simulation runs, and wires the agent to consume them. Use when the user asks "what should I test", "generate simulation scenarios", "write scenarios for my agent", "add a scenario for X", "organize my scenario files", "my simulations are flaky", "the scenario hits my real database", "seed state per scenario", "it passed but booked the wrong thing", "grade the final state", or wants to stress-test a flow before shipping. Covers generating a baseline with the LiveKit scenario generator and refining it, writing the cases generation misses, phrasing simulated-user instructions so they steer reliably, splitting scenarios into sets across files, and the agent-side code that seeds deterministic state from a scenario and fails a run on its final state. To run them use running-livekit-simulations.'
license: MIT
metadata:
  author: livekit
---

# Writing simulation scenarios

A scenario is a simulated user's **`instructions`** (who they are, what they want) plus
**`agent_expectations`** (what counts as success). A simulation plays the scenario against the real
agent, and an LLM judge grades the transcript.

Runs are disposable. The scenario file is what lasts: it gets reviewed in diffs, re-run for years,
and it's what catches a prompt edit that breaks something without anyone noticing.

Before writing a file, look up the exact schema and CLI flags with `reading-livekit-docs`. Field
names and commands change, so this skill doesn't restate them. Conceptually, a file is a named group
of scenarios. Each scenario has the simulated user's instructions, the pass criteria, and optionally
tags for grouping and per-scenario data the agent can read at runtime. If the CLI offers to add
stable per-scenario ids to a file, accept and commit them.

## Start from a generated baseline

Don't start from an empty file. LiveKit's generator reads the agent's source and produces a decent
first pass much faster than you'd write ten scenarios by hand:

```bash
lk agent simulate text -n 10    # confirm the exact flags with --help
```

**Generating from source uploads the code to LiveKit Cloud** so the generator can read it, which is
why the CLI asks for confirmation first. Tell the user that before running it and let them decline.
Some code can't leave the machine, and in that case you write the scenarios by hand. A flag skips
the prompt for non-interactive runs.

When the run finishes, the CLI either tells you where it saved the generated scenarios or offers to
save them into the project. Either way, get that file into the repo as the starting point.

You can't steer the generator. It infers intent from the code, so it writes plausible conversations
instead of the ones this agent's users have, and it doesn't know what the user is worried about.
Treat its output as a draft:

1. **Read every scenario** and delete the ones that don't matter. A file you haven't read isn't a
   test suite.
2. **Sharpen vague expectations.** If the judge can't decide an `agent_expectations`, its verdict
   flips between runs. This is the biggest cause of flaky scenarios.
3. **Add what generation misses.** It drifts toward happy paths. `references/risk-coverage.md`
   explains how to turn the agent's constraints into a checklist and give each item a scenario.
4. **Add what the user is worried about.** If they haven't said, ask what they want stress-tested.
   That input is why a suite a person steers is better than a generated one.

## Ask what to probe

The user knows which flow keeps breaking, which customer complained, and which change they're
nervous about. Ask, and weight the suite toward that area with more and deeper scenarios.

Focus adds coverage. It never removes coverage of the agent's hard limits. If the user has no
preference, generate broadly and tell them that's what you did.

## Ground scenarios in what the agent can do

Read the agent's code locally with your normal tools before writing anything. Look for what a user
can ask for (**capabilities**), where requests get blocked (**constraints**: required steps,
unavailable items by name, caps, eligibility), and what the agent must refuse.

Constraints matter most. A scenario that asks for something the agent can't do is only valid if the
expectation is that the agent *says so*. Written the other way, the test fails when the agent
behaves correctly.

Scope is the agent reachable from the session entrypoint, plus agents it hands off to and tasks it
awaits. Ignore other classes in the directory, unused imports, and example files.

## Write instructions the simulated user can follow

There are two shapes, and which one you use depends on what the scenario is for. Details and
examples are in `references/scenario-craft.md`.

- **Point-form direction** for deterministic flows you'll grade on end state. A structured template
  (persona, opening line, facts revealed only when asked, steps in order, conditional reactions)
  keeps the persona model on track and makes the run repeatable.
- **A persona paragraph plus goals** for open-ended and adversarial scenarios graded only on the
  conversation, where you want the simulated user to improvise.

In both shapes, goals are requests *to* the agent, never the agent's own actions. Use real values
from the agent's domain and assume no prior state. Vary persona, mood and difficulty across the
suite so it isn't ten copies of the same cooperative caller.

## Split scenarios into sets, one file per set

The run command takes one scenario file, so **a file is a run**. Split along the lines you want to
run separately, which usually isn't topic.

The most important axis is **how the set is graded**:

- Scenarios that drive concrete tool flows to a deterministic end state, graded on final state *and*
  the conversation (see "Make the agent consume the scenario" below).
- Open-ended and adversarial scenarios graded on `agent_expectations` alone.

These need different instructions shapes, different agent wiring, and often a different run
cadence, so they go in different files. The second axis is **cadence**: a small set to run before a
merge, and the full set before a release.

Give each file a `name` that describes the set, since it labels the run. Inside a file, use `tags`
to slice. Tag `feature` so a failure points at the part of the agent that owns it, and add whatever
else you filter by (channel, difficulty). Put a header comment on the file recording the set's
assumptions: pinned dates, required environment variables, and what its expectations depend on.

## Keep scenarios reproducible

A scenario should give the same result months from now as it does today.

- **Write absolute dates** and pin the agent's clock (usually with an environment variable) so
  availability and expectations always line up. Relative dates rot.
- **Don't let scenarios hit real backends.** Seed deterministic state from the scenario instead
  (see "Make the agent consume the scenario" below).
- **Keep ids stable** where the CLI supports them. Labels and instructions change, and the id is
  what ties a scenario's runs together over time. A scenario rewritten from scratch is a new
  scenario and gets a new id.

## Make the agent consume the scenario

A scenario file can't fix two problems by itself. A scenario that reaches a real backend grades
differently every run, and a judge that only reads the transcript will pass a run that booked the
wrong room. Both are fixed in the agent's code, in one branch at the top of the entrypoint:

1. **Detect** a simulation from the job context. In production the check comes back empty and
   nothing else changes.
2. **Seed** state from the scenario's per-scenario data (a fake calendar, an in-memory database)
   that the agent's real tools read through unchanged.
3. **Mock tools for the session's lifetime**, so tool selection is still under test but execution
   is deterministic.
4. **Grade the final state** in the end-of-simulation callback. Compare what the agent ended with
   against what the scenario expected, and fail the run if they differ. Your check can fail a run
   the judge passed. It can't rescue one the judge failed.

Only tool-flow scenarios need this. Open-ended and adversarial ones are graded on the conversation
alone. `references/connecting-the-agent.md` covers it in full, including how to confirm the wiring
took effect. Look up the current API names with `reading-livekit-docs`.

## Grow the suite from real failures

The most valuable scenarios describe something that went wrong. Once the agent is live, derive them
from recorded sessions instead of inventing more. Where the CLI supports it, a subcommand derives a
scenario from a recorded session (check `lk agent simulate --help`). A derived scenario describes
one call, so widen it to the class of calls it represents and sharpen its expectation into the rule
you want enforced.

Every production bug you fix should get a permanent scenario in the file.

## Don't write bad tests

The judge grades the agent against `agent_expectations`, so a careless expectation punishes correct
behavior:

- For guardrails and negative cases, the pass is that the agent **refuses, escalates, or declines to
  invent data**. Write the expectation that way.
- Never write an expectation the agent can only meet by misbehaving, such as stating data it can't
  know or giving specific medical, legal or financial directives. If passing requires misbehavior,
  the scenario is wrong.
- Judge by outcome, from the user's point of view. Don't prescribe wording.

## References

- `references/risk-coverage.md` — turning constraints into a checklist and guaranteeing coverage
- `references/scenario-craft.md` — instructions shapes, persona variety, audio-specific scenarios
- `references/connecting-the-agent.md` — the agent-side code: detect, seed, mock, grade final state

## Related skills

- Running them: `running-livekit-simulations`
- Schema and CLI facts: `reading-livekit-docs`
