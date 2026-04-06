# GitHub Issue Templates for livekit-agents

These templates mirror `.github/ISSUE_TEMPLATE/` in the repository.

---

## Bug Report Template

**Labels:** `bug`

### Required fields

| Field | Notes |
|---|---|
| `Bug Description` | Clear description of the bug and current (broken) behavior |
| `Expected Behavior` | What the correct behavior should be |
| `Reproduction Steps` | Numbered steps; include a minimal code snippet or GitHub Gist |
| `Operating System` | e.g. macOS Tahoe, Ubuntu 22.04 |
| `Package Versions` | `livekit==`, `livekit-agents==`, `livekit-api==`, plus any plugin versions |

### Optional fields

| Field | Notes |
|---|---|
| `Models Used` | STT/LLM/TTS setup, e.g. "Deepgram Nova-3, OpenAI GPT-4.1, Cartesia Sonic-2" |
| `Session/Room/Call IDs` | LiveKit Cloud IDs: `roomID: RM_`, `SIP Call ID: SCL_`, `sessionID:` |
| `Proposed Solution` | Code snippet or description of the fix (Python code block) |
| `Additional Context` | Logs, stack traces, related issues |
| `Screenshots and Recordings` | Links or attachments |

### Markdown structure for draft file

```markdown
## Bug Description

<clear description of current broken behavior>

## Expected Behavior

<what should happen>

## Reproduction Steps

1.
2.
3.

\`\`\`python
# minimal reproducible example
\`\`\`

## Environment Details

- **OS:**
- **Models Used:**
- **Package Versions:**
  \`\`\`
  livekit==
  livekit-agents==
  livekit-plugins-openai==  # etc.
  \`\`\`

## Session / Room / Call IDs

(if applicable)

## Proposed Solution

\`\`\`python
# suggested fix
\`\`\`

## Additional Context

<logs, traces, links to related code>
```

---

## Feature Request Template

**Labels:** `enhancement`

### Required fields

| Field | Notes |
|---|---|
| `Feature Type` | One of: "Nice to have" / "Would make my life easier" / "I cannot use LiveKit without it" |
| `Feature Description` | Clear description of the desired capability and why it's needed |

### Optional fields

| Field | Notes |
|---|---|
| `Workarounds / Alternatives` | Current workarounds the user has tried or considered |
| `Additional Context` | Screenshots, links, related issues |

### Markdown structure for draft file

```markdown
## Feature Type

**Would make my life easier** (or: Nice to have / I cannot use LiveKit without it)

## Feature Description

<what you want to happen and why — include the motivation/use case>

## Workarounds / Alternatives

<any current workarounds or alternative approaches you have considered>

## Additional Context

<links to related issues, code examples, etc.>
```

---

## Tips for Good Issues

- **Bug titles**: start with the component in parentheses, e.g. `(stt): FallbackAdapter leaks stream on provider timeout`
- **Feature titles**: use an imperative verb, e.g. `Add option to configure silence threshold per-session`
- **Code references**: always include file path and line number when referencing source, e.g. `livekit-agents/livekit/agents/stt/fallback_adapter.py:L142`
- **Minimal reproduction**: strip the example down to the fewest lines that still trigger the issue
- **Version pinning**: always list exact versions in bug reports, not ranges
