# Writing instructions a simulated user follows well

## Contents
- Two shapes: point-form direction, and persona-and-goals
- Rules that hold for both shapes
- Varying the cast across a suite
- Identity details in instructions
- Scenarios written for audio

The simulated user is an LLM that reads `instructions` and improvises a conversation. How you write
them decides whether a run is repeatable or a different conversation every time, and the right
approach depends on what the scenario is for.

## Two shapes

### Point-form direction — for deterministic flows

When a scenario drives a concrete flow to a known end state and you grade on that state, the
simulated user has to stay on track. A structured, labelled template steers much more reliably than
prose:

```
PERSONA: who they are, and any speech detail that matters (spelling a name only if asked)
OPENING LINE: the exact first thing they say
FACTS (reveal each only when asked, one per turn):
  - the details the agent must collect
DO, IN ORDER:
  - the steps they take, in sequence
REACTIONS:
  - conditional behavior: if the agent offers X, decline; if it asks twice, get impatient
HIDDEN TRUTH: what's actually true, which the agent must discover
```

Two parts of this matter most. **Revealing facts one per turn, only when asked** makes the agent run
its collection flow instead of getting everything in the opening line. **Steps in order** keep the
path stable across runs, which is what makes final-state grading meaningful.

When a scenario grades on final state, repeat the pinned values (dates, contact details, options)
verbatim in both the instructions and the expected state so the comparison can match.

### Persona and goals — for open-ended and adversarial scenarios

When you're grading the conversation itself and want the simulated user to improvise, too much
direction defeats the purpose. Write a short persona in the third person (communication style and
mood), then a handful of atomic goals:

```
A brisk regular who has done this before and doesn't want small talk.

Goals:
- Ask what's available tomorrow afternoon
- Change their mind about the time once the agent offers something
- Finish with a confirmed booking
```

## Rules that hold for both shapes

- **The simulated user talks *to* the agent.** Goals are requests ("order a large fries"), never the
  agent's own actions ("greet the caller", "process the order"). A simulated user doing the agent's
  job is the most common broken scenario.
- **Only what the agent handles.** Base every goal on a capability the agent has. Don't ask a
  drive-thru for delivery unless you're testing that refusal, and in that case say so in the
  expectation.
- **Real domain values.** Use the item names, sizes and times the agent works with.
- **No prior state.** Each goal reads independently. Write "Order a Big Mac" then "remove the Big
  Mac", never "remove the Big Mac that was already added".
- **Mix difficulty.** Mostly straightforward, some with a change of mind partway through, a few
  hard ones.

## Vary the cast

Give each scenario a distinct persona, mood and situation, and spread them widely: age, tone, tech
comfort, urgency, patience, how cooperative they are. Keep their circumstances consistent with how
they'd plausibly end up talking to this agent.

Spread the suite across framings so it doesn't all sound the same:

- **Routine**: the everyday request the agent handles constantly.
- **Common but characterful**: a normal request made interesting by who's making it.
- **Uncommon but plausible**: realistic but not typical.
- **Stress**: difficult but realistic behavior for this domain.

For stress cases, give the caller a **communication challenge**: hostile, evasive, won't verify,
tries to befriend the agent, talks around the question. The difficulty should be in *how* they talk
while the request itself stays realistic. An absurd request tests nothing. A reasonable caller who
won't answer questions tests a lot.

Also cover the agent's whole range. Don't only test the first or most popular item.

## Identity details

Put the identity the scenario needs directly in the instructions (name, contact details, payment
details where the flow needs them), using obviously fictional values. Keep them consistent with any
seeded state and with the expected end state.

It's fine, and often the point, for a caller to *lack* a credential: no order number, no PIN, can't
verify. Don't invent a specific wrong value unless the scenario is about a mismatch.

## Scenarios written for audio

Most scenarios should run in text. Write a scenario *for* audio when what you're testing only
exists in speech:

- **Entities that have to survive transcription**: spelled-out names, confirmation codes, account
  numbers, amounts, addresses. Put them in the instructions and require them back correctly.
- **Turn-taking and interruption**: a caller who interrupts mid-sentence, trails off, pauses to
  think, or answers before the agent finishes.
- **Recovery from bad input**: a caller on a poor connection or in a noisy place, where the agent
  should ask them to repeat instead of guessing.

Put these in a set of their own. Audio runs are slow and expensive, so you want to choose when to
run them instead of pushing the whole file through the audio pipeline.
