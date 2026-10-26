---
name: reading-livekit-docs
description: 'Looks up current LiveKit facts (API signatures, CLI flags, config options, model and provider support, SDK changelogs, pricing) from the docs instead of answering from memory. Use whenever a question touches LiveKit specifics: "does LiveKit support X", "what changed in agents 1.8", "what are the arguments to Y", "how much does LiveKit cost", "find an example of Z in the LiveKit repos", or before writing any LiveKit code. Other LiveKit skills load this one first. Covers the LiveKit Docs MCP server and the `lk docs` CLI.'
license: MIT
metadata:
  author: livekit
---

# Reading LiveKit documentation

LiveKit's SDKs change faster than model training data, so anything you remember about a LiveKit
API, CLI flag, or default is a guess. Wrong guesses cost time: a wrong flag wastes a run, a wrong
signature wastes a build.

**Every LiveKit-specific fact you state or put in code should come from a lookup in this
session.** That includes facts you're confident about and patterns you've seen in other repos.

## Where to look, in order of preference

**1. The LiveKit Docs MCP server.** If its tools are in your session, use them. They're the same
source the CLI uses, with less friction.

**2. `lk docs`.** Available wherever the LiveKit CLI is installed, with no MCP setup. Use it when
the MCP tools aren't there; don't stop to ask the user to install anything first. It can search the
docs, fetch pages, search code across LiveKit's repositories, show an SDK's recent changelog, report
pricing, and submit feedback. Start with:

```bash
lk docs --help
lk docs search "agent simulations"
```

Once search has shown you the right page, fetch it. There's machine-readable output if you'd
rather parse than read.

**3. Web search against `docs.livekit.io`.** Only when neither of the above is available. Tell the
user you fell back to it, and mark generated code as unverified.

## How to research

- **Search before you read.** Search returns per-page excerpts, which is usually enough to pick the
  right page. Fetch full pages only once you know which one you need, or you'll fill the context
  window with pages you don't use.
- **Use code search for how, docs for what.** Docs tell you a parameter exists. LiveKit's own
  examples show how it's called. When a docs page and a working example disagree, go with the
  example and mention the discrepancy.
- **Use the changelog for version questions.** "Is X available yet", "when did Y change", and "why
  doesn't this argument exist" are answered by the changelog, not search.
- **Check the version you're on.** The installed SDK and CLI determine what works, and a documented
  feature may not exist in the installed version. For any CLI command, `--help` on the installed
  binary outranks every other source.

## Reporting what you found

- **Cite the page** when a decision rests on it, so the user can check you.
- **Say when you couldn't verify something.** "I could not confirm this signature against the docs"
  is useful to the user. A silent guess isn't.
- **Mark unverified code.** If you had to write LiveKit code without a lookup, add a comment at the
  call site saying it's unverified, and say so in your reply.
- **Don't invent a flag or parameter to make an example look complete.** An incomplete example with
  a note is better than a plausible fabrication.

## When the docs are wrong

Sometimes a page contradicts the SDK source or the CLI's `--help`. When that happens:

- **Go with what runs.** `--help`, the installed SDK's source, and a working example outrank a
  prose page. Tell the user which one you followed and why.
- **Report it** with the docs feedback command, and tell the user you did, so the next person
  doesn't hit the same error.

`lk docs` sometimes warns about version skew when the docs server is a little ahead of the CLI.
That's routine and the results are still good. Suggest updating `lk` and keep going.

## Related skills

- Building an agent: `building-livekit-agents`
- Live-testing one during development: `debugging-livekit-agents`
- Unit tests: `testing-livekit-agents`
- Simulations: `writing-livekit-scenarios`, `running-livekit-simulations`
