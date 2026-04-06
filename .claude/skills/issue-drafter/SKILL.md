---
name: issue-drafter
description: Draft GitHub issues for the livekit-agents repository. Use this skill when asked to investigate code and draft a bug report or feature request, check whether an issue is worth filing, or write an issue draft to a local markdown file. Triggers on requests like "draft an issue", "file a bug", "create a feature request", "write up an issue", "調査してissueを立てる", or when asked to investigate a specific problem and document it.
---

# Issue Drafter

Draft well-structured GitHub issues for the livekit-agents repository by investigating the codebase, determining whether filing an issue is warranted, and writing an English-language draft to a local markdown file.

## Workflow

Follow these steps in order:

### Step 1: Investigate the Code

Before drafting anything, do a thorough code investigation:

1. Identify the relevant files/modules from the user's description
2. Read the relevant source code carefully — understand current behavior vs. expected behavior
3. Check git log for recent changes to the relevant area (`git log --oneline -10 -- <path>`)
4. Look for related tests, existing issue hints in comments, or TODO markers

**Judgment gate — decide whether an issue is genuinely warranted:**

- **File a bug report** if: there is a reproducible unexpected behavior, a crash, incorrect output, or a violation of documented behavior
- **File a feature request** if: a useful capability is clearly missing, or the current API is unnecessarily limiting without a reasonable workaround
- **Do NOT file an issue** if: the behavior is intentional, already documented, trivially fixable by the caller, or likely covered by an existing known issue — explain the reasoning to the user instead

### Step 2: Select the Template

Read `references/templates.md` to pick the right template based on issue type:
- **Bug** → Bug Report
- **Feature / Enhancement** → Feature Request

### Step 3: Draft the Issue in English

Write the issue following the selected template. Be specific and technical:

- Include concrete code references (file paths, line numbers, function names)
- Describe actual vs. expected behavior precisely
- Propose a solution or implementation approach where possible
- Keep the title concise and actionable (≤ 72 characters)

### Step 4: Write Draft to File

Save the draft to a local markdown file:

- Path: `./issue-drafts/<slug>.md` relative to the repository root
- Slug: lowercase, hyphens, descriptive (e.g., `stt-fallback-stream-leak.md`)
- Create `./issue-drafts/` directory if it does not exist

## Output File Format

```markdown
---
type: bug | feature
title: "<issue title>"
labels: [bug | enhancement]
template: bug_report | feature_request
---

## Bug Description
...

## Expected Behavior
...

## Reproduction Steps
...

## Environment Details
...

## Proposed Solution
...

## Additional Context
...
```

For feature requests, replace bug-specific sections (Expected Behavior, Reproduction Steps, Environment Details) with:
- **Feature Type** (Nice to have / Would make my life easier / Critical)
- **Workarounds / Alternatives**

## References

- **Template field details**: See `references/templates.md` — read this before filling out any template to ensure required fields are covered correctly
