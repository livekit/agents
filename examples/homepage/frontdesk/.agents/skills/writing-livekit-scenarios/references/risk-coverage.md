# Guaranteeing coverage of an agent's hard cases

Scenario generation, whether yours or the LiveKit generator's, drifts toward plausible happy paths.
Left alone, it produces ten cooperative callers who get what they want and skips the cases that
break agents. Persona variety doesn't fix this. Ten different personalities all making easy
requests still test one thing.

So don't rely on a good prompt to produce coverage. Make an explicit checklist and work through it.

## Build the checklist

Read the agent's code. For every hard limit you find, write one line with an id, a category, and
what must be tested. Keep it in a file (`risks.yaml` or similar) next to your working notes.

Where entries come from, roughly in order of how often they're missed:

- **Unavailable items, by name.** Every item, plan or option the agent can't currently provide gets
  its own entry. A general rule like "it won't offer things that are unavailable" can't be tested,
  so it isn't an entry.
- **Required inputs.** Anything the agent must collect before it can proceed.
- **Invalid values.** Every rule on what a value may be: caps, supported options, time windows,
  eligibility.
- **Preconditions.** Steps that can't be skipped, and gated actions.
- **Empty results.** Lookups that can legitimately return nothing.
- **Guardrails.** Out-of-scope requests, harmful requests, requests for professional advice,
  sensitive data handling, attempts to extract the system prompt.

Guardrails are the part people forget, and a guardrail failure is the one that ends up in a
screenshot.

## Every entry gets a dedicated scenario

A passing mention doesn't count. Each entry needs a scenario where the simulated user *does the
thing*. Asking "do you have X?" doesn't test that X is unavailable. Ordering X does. What the
simulated user does for each category:

| Category | What the simulated user must do |
|---|---|
| unavailable | Order or request the unavailable item *by name*, and press when told no |
| withhold-required | Proceed through the flow but refuse or omit the required input when asked |
| invalid-value | Supply a value that breaks the rule: an off-list option, over-limit amount, unsupported plan |
| precondition | Try to get the gated action done while skipping the required step |
| resource-empty | Drive the lookup to nothing available, then press for a result anyway |
| out-of-scope | Ask for something the agent doesn't do |
| harmful | Make an unsafe or disallowed request |
| professional-advice | Ask for a *specific* medical, legal or financial recommendation, not general information |
| sensitive-data | Volunteer or demand handling of a full card number, government id, password, or someone else's record |
| prompt-extraction | Ask the agent to reveal its instructions, or tell it to ignore them |

In every one of these, **the pass is the agent refusing, declining to make something up, or stating
the limit honestly.** Write `agent_expectations` that way. Otherwise the test fails when the agent
behaves correctly.

## Check coverage before you run

Before running, list each checklist id next to the scenario label that covers it. Any id without a
scenario gets one. If you're at your budget, replace a redundant happy-path scenario instead of
letting the file grow.

Tag scenarios by what they cover (`tags` works well) so the mapping is recorded in the file and not
only in your head.

## Budget

Coverage is a minimum. Forty scenarios nobody reads are worse than twelve that each pin down
something specific. Every scenario costs a full conversation to run, and an unread scenario with a
vague expectation gives flaky verdicts that make people stop trusting the whole file.

A rough order: the paths that carry most of the traffic, the edges of each of those paths, then one
per checklist entry. Add the user's stated worries on top, and stop there.
