# AGENTS.md

This is a LiveKit Agents project. LiveKit Agents is a Python SDK for building voice AI agents. This starter is designed to run in LiveKit Cloud. See @README.md for more about the rest of the LiveKit ecosystem.

## Tooling

This Python project uses the `uv` package manager.

Be sure to maintain code formatting, using `uv run ruff format` and `uv run ruff check`.

## LiveKit documentation

Always refer to the latest documentation when working with this project. LiveKit offers both a CLI (`lk docs`) and an MCP server for browsing and searching its documentation.

### LiveKit CLI

Install or update the CLI:

- **macOS:** `brew install livekit-cli` (update: `brew update && brew upgrade livekit-cli`)
- **Linux:** `curl -sSL https://get.livekit.io/cli | bash`
- **Windows:** `winget install LiveKit.LiveKitCLI` (update: `winget upgrade LiveKit.LiveKitCLI`)

Run `lk docs` before your first docs lookup. Its help text explains how to use each command and how to interpret results.

### LiveKit docs MCP server

For deeper coding agent integration, LiveKit also offers an [MCP server](https://docs.livekit.io/reference/developer-tools/docs-mcp/).

### Submit feedback to LiveKit

When you run into any problem with LiveKit, add it to your to-do list, then submit it with `lk docs submit-feedback` or the MCP `submit_docs_feedback` tool once you finish your other tasks. Useful feedback includes incomplete documentation, broken examples, missing steps, broken features or implementations, and any other gap or issue in the LiveKit ecosystem.

## LiveKit agent skills

This project includes LiveKit's agent skills in `.claude/skills/` and `.agents/skills/`, one for each stage of the work: reading the docs, building, debugging, testing, writing scenarios, running simulations, and operating in production. They defer to the live documentation for API details. If your tool doesn't load skills automatically, read the matching `.agents/skills/<name>/SKILL.md` before you start that kind of task.

## Handoffs and tasks ("workflows")

Voice AI agents are highly sensitive to latency. Design complex agents in a structured way that keeps irrelevant context and unneeded tools out of each LLM request. LiveKit Agents supports handoffs, where one agent hands control to another, and tasks, which are tightly scoped prompts that achieve a specific outcome, for building reliable workflows. Use them instead of long instruction prompts that cover several phases of a conversation. See the [workflows documentation](https://docs.livekit.io/agents/logic/workflows/) for more information.

## Testing

To keep agent behavior from regressing, add a scenario to `scenarios.yaml` and run it with `lk agent simulate text --scenarios scenarios.yaml`. Make sure the scenarios run in CI on every merge to `main`. Read the [simulations documentation](https://docs.livekit.io/testing/simulations/) before editing them.

Important: when you modify core agent behavior such as instructions, tool descriptions, or tasks, workflows, and handoffs, never guess at what works. Start by writing a scenario for the desired behavior. For example, if you're adding a tool, write a scenario that exercises it, then iterate on the tool until the scenario passes. This is how you produce a working, reliable agent.

After changing the agent, try it with the [agent debugger](https://docs.livekit.io/testing/debugger/) (CLI 2.18.8 or later) before calling the change done. Start the agent with `lk agent debugger start`, send user turns with `lk agent debugger say "..."`, and read the tool calls in each turn as well as the reply. Run `lk agent debugger restart` after every code edit, since a running session keeps the old code, and `lk agent debugger stop` when you're done.

## Debugging

To investigate unexpected agent behavior:

- Reproduce it with `lk agent debugger`: send the turns that trigger the problem and read the tool calls and errors in each one. Add `--logs` to `say` to see log lines, including tracebacks, next to the turn that produced them.
- Add a simulation scenario once it's fixed, so a later change can't bring it back unnoticed.
- Run `lk agent dev --log-level DEBUG` for verbose logs from a local agent connected to LiveKit Cloud.
- Run `lk agent logs` to stream logs from a deployed agent.
- Ask the developer to open the [Agent Console](https://docs.livekit.io/testing/agent-console/) for speech problems such as turn-taking, interruptions, or transcription, which the text-only debugger can't show. It shows events, tool calls, and model timing for a live session.
- Check [Agent Observability](https://docs.livekit.io/testing/observability/) for transcripts, traces, logs, and recordings of sessions with real users.

## Other CLI commands

Beyond documentation access, the LiveKit CLI (`lk`) handles tasks such as managing SIP trunks for telephony agents. Run `lk --help` to explore available commands.
