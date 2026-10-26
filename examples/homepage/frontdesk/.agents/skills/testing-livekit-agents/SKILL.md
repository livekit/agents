---
name: testing-livekit-agents
description: 'Writes turn-level tests for a LiveKit agent in the user''s normal test suite: pytest (Python) or Vitest (Node.js). Use when the user asks to "write tests for my agent", "add a test for this tool", "test the handoff", "pin this bug", "why does my agent test fail", or after building or changing agent behavior that needs regression coverage. Covers the SDK''s test session harness, assertions on messages, tool calls and handoffs, LLM judging of a reply against an intent, mocking tools, multi-turn tests, and judging whole conversations with the built-in judges. For interactive poking use debugging-livekit-agents. For grading whole conversations at scale use running-livekit-simulations.'
license: MIT
metadata:
  author: livekit
---

# Testing LiveKit agents

Turn-level tests are the cheapest lasting verification an agent can have. They run in the user's
existing suite, in text mode, and they're fast enough for every commit. The framework's helper
names and signatures change, so look them up with `reading-livekit-docs` before writing, and use
the testing docs page for every API detail this skill leaves out.

Every test has the same shape: start a test session with the agent under test, run one user turn,
and assert on the events that turn produced.

## What a test asserts on

A turn produces a sequence of events. A simple turn is one message. A more typical one is a tool
call, its output, maybe a handoff, and then a message. You write the test by walking that sequence
in order, asserting on each event, and then asserting the turn has nothing more.

There are three kinds of assertion, and most of this skill is knowing which one to use.

**Structural** assertions check message roles, that a tool was called, its arguments, what it
returned, and that a handoff to a specific agent happened. They're deterministic and fail for
exactly one reason, so prefer them.

**Judged** assertions use an LLM-judge helper that gives one message and an intent string to a
model and asks whether they match. Use them for the content of a reply, which you can't assert
exactly. Describe the intent by outcome ("tells the user the booking is confirmed and gives the
time"), not by wording.

**Whole-conversation** assertions use a judge-group helper that runs several built-in judges
concurrently over the whole chat history and aggregates their verdicts. The built-in judges cover
dimensions like grounding, relevance, safety, task completion, and tool use; the docs list the
current set. Use this when the question spans several turns.

Rules of thumb:

- **Assert structure first and judge only what's left.** A judged assertion that could have been
  structural is slower and flakier.
- **Close the turn.** Assert there are no further events. Otherwise a test can pass while the agent
  also does something you didn't intend.
- **One behavior per test.** When a test that asserts six things fails, it tells you very little.

## Mocking tools

Tests shouldn't hit real backends, since that makes a suite slow and nondeterministic. Override the
tools for the agent under test and return fixed values.

Two things beyond the API:

- **Mock failures as well as successes.** Returning an error from a mock makes the tool raise, which
  lets you test what the agent says when a backend is down. Most agents lack this test, and it's
  the behavior users notice most.
- **Scope matters.** By default, mocks apply in a block around your own `run()` calls, which is what
  a test needs. A session-scoped form also exists for a session that runs on its own and needs
  mocks active for its whole lifetime. Simulation entrypoints use that form, tests don't. See
  `writing-livekit-scenarios`.

Mocking only changes execution. The model still sees the real tool schemas, so tool selection is
still under test.

## Multi-turn and seeded history

There are two ways to test behavior that depends on earlier turns:

- **Run the turns.** Each turn adds to the conversation history, so the second turn sees the first.
  Use this when the path matters: collecting details across turns, the user changing their mind
  mid-flow, a handoff carrying context.
- **Seed the history.** Build a chat history and give it to the agent to start at the state you
  care about. Use this when the setup turns aren't what you're testing. It's faster and less brittle
  than replaying five turns to reach the sixth.

## What to test

Roughly in order of value:

1. **The behavior the user asked for.** Every agent needs at least this.
2. **Tool invocation**: the right tool with the right arguments for a representative request.
3. **Tool failure**: what the agent says when a tool errors or returns nothing.
4. **Refusals and limits**: the agent declines what it should decline and doesn't make up data it
   can't have. The refusal is the pass condition.
5. **Handoffs**: the transition fires when it should, and the next agent has what it needs.
6. **Every bug you've fixed.** Without a test, a fixed bug can come back unnoticed.

## Three kinds of evidence, kept distinct

A test suite for an agent that does things — books, edits, confirms — needs three kinds of test,
and confusing them is how a green suite ships a broken agent:

- **Deterministic application tests** prove guards, no-ops, empty-vs-unknown values, corrections,
  storage failure, and retries, against the application core directly. Fast, exact, no model.
- **Real-SDK tests with a scripted provider** prove event wiring and tool plumbing: that the
  listener fires, that the tool receives what the session sends. A scripted provider proves
  *mechanics*, never language understanding — its exact sentences are fixtures, not requirements.
- **Ordinary-language tests with the real model** prove the model can complete the task from how
  callers actually talk: paraphrases absent from the prompt, corrections mixed with agreement,
  the same short reply answering different questions.

Shortcuts that look like tests and aren't: calling a helper or filling private state instead of
driving the interaction; telling the test caller which tool to name; mocking the very mutation
whose correctness is under test; comparing output only with the application's own exporter;
retrying a failed turn until it passes; deleting the assertion that failed.

## Don't write tests that punish correct behavior

The most common bad agent test asserts that the agent does something it shouldn't: states data it
can't know, gives a specific medical, legal, or financial recommendation, or completes a flow that
should have been blocked. If the agent can only pass by misbehaving, fix the test.

For a guardrail, the pass is that the agent refuses, escalates, or declines to make something up.
Write the assertion that way.

## Where this sits

- **Interactive poking while building** → `debugging-livekit-agents`. Faster loop, no assertions kept.
- **Turn-level, deterministic, every commit** → here. Cheapest lasting check.
- **Whole conversations graded by a simulated user, before a release** → `running-livekit-simulations`.
  More expensive, and catches emergent behavior that turn-level tests miss.

When a simulation keeps failing the same way, the bug is usually turn-level. Write a test for it
here, which pins down the cause more precisely and catches it earlier.

## Related skills

- Current API surface: `reading-livekit-docs`
- Finding the bug first: `debugging-livekit-agents`
- Simulations, including seeded state and session-scoped mocks: `writing-livekit-scenarios`, `running-livekit-simulations`
