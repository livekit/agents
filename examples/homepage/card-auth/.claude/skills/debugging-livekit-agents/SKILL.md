---
name: debugging-livekit-agents
description: 'Drives a multi-turn conversation with a LiveKit agent running locally to see what it does. Use when the user says "test my agent", "try my agent", "does this work", "why did it call that tool", "it says the wrong thing when I ask X", "test this change", or whenever you have edited an agent and need to check how it behaves. Wraps `lk agent debugger`: start the agent in text mode, send turns, read the tool calls, handoffs, errors and logs behind each reply, and restart after an edit. It runs without audio or a LiveKit room at one LLM call per turn, so it is the preferred way for a coding agent to live-test during development, and the default when the user says "test" without naming unit tests or simulations.'
license: MIT
metadata:
  author: livekit
---

# Debugging a LiveKit agent live

`lk agent debugger` runs the user's agent locally as a background process in text mode and lets
you drive a conversation one turn at a time. It's built for coding agents: you play the user,
choose each next line based on the last reply, and inspect what the agent did in between.

Speech is off and nothing goes to a LiveKit room, so a turn costs only the agent's own LLM and tool
calls. That's cheap enough to use constantly while building.

Before the first use, confirm the command exists and read its help:

```bash
lk agent debugger --help
```

The help is thorough and is the source of truth for subcommands and flags, so this skill doesn't
restate them. If the command is missing, the installed CLI predates it. Tell the user to update
`lk` and use `testing-livekit-agents` until then. Don't guess at an older command's shape.

## The loop

**Start** the agent, **say** user turns, **inspect** what happened, edit the code, **restart**,
repeat, and **stop** when you're done. Roughly:

```bash
lk agent debugger start
lk agent debugger say "Hi, what can you do?"
lk agent debugger say "Book me a table for two tonight"
lk agent debugger stop
```

Each `say` prints everything the agent did in response (tool calls with arguments and results,
handoffs, errors) followed by the reply. Between turns you can look at the agent's chat history, a
live event stream, the process logs, and status; `--help` lists the subcommands.

**Restart after every code edit.** A running session keeps the old code, and it's easy to lose
time debugging behavior the file no longer has.

## Debugging with it

- **Read the tool calls as well as the reply.** The reply shows what went wrong; the tool call and
  its arguments usually show why. A wrong argument points to the prompt or the tool description. No
  call at all usually means the tool description never says when to use it.
- **Interleave the logs when a tool misbehaves.** An option shows the agent's log lines under the
  turn they belong to, so a tool's traceback appears right below the sanitized error the user would
  have heard. That's the quickest way from symptom to cause.
- **Check the agent's chat history instead of relying on your memory of the conversation.** It
  records what the LLM saw, including instruction and tool changes across handoffs. When behavior
  seems impossible, the history usually shows the context isn't what you assumed.
- **Reproduce before you fix, then re-run the same turns.** Keep the sequence of `say` lines that
  triggered the bug so you can compare before and after.
- **Drive whole conversations.** Most bugs take several turns to show up: details collected early
  that get lost, a user changing their mind mid-flow, a handoff that drops context. A single turn
  won't find them.
- **Script it if a program is deciding the turns.** There's machine-readable output and meaningful
  exit codes. Check `--help` for the current format instead of assuming field names.

## When to use something else

| Situation | Use |
|---|---|
| You want to *hear* it, or hand the user something to try | `lk agent console` (mic and speakers, a human at the keyboard) |
| The same failure keeps coming back | A turn-level regression test (`testing-livekit-agents`) |
| You need whole-conversation outcomes graded at scale | Simulations (`running-livekit-simulations`) |
| The bug is about speech: turn-taking, interruptions, transcription | Audio simulations. The debugger is text-only and can't see these |

The debugger is for finding bugs interactively. Once you've found one, write a test for it so a
later change can't reintroduce it unnoticed.

## Related skills

- Building the agent: `building-livekit-agents`
- Pinning a found bug as a test: `testing-livekit-agents`
- Whole-conversation checks: `writing-livekit-scenarios`, `running-livekit-simulations`
- Production-only failures (worker processes, providers, shutdown): `operating-livekit-agents`
- CLI flags and API facts: `reading-livekit-docs`
