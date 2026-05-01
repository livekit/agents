---
name: pr-description
description: Generate or update PR descriptions following LiveKit conventions
---

Create a pull request description for this branch. If updating, preserve key information.

## Format

```markdown
## Summary

- Clear, bullet-point overview of changes
- Focus on the _why_, not just the _what_

## Changes

- List specific files/modules affected
- Note any breaking changes with ⚠️ prefix

## Testing

- How was this tested?
- Include commands or steps if applicable

## Related

- Fixes #XXX (if applicable)
```

## Guidelines

- **Be concise** - Reviewers should understand in 30 seconds
- **Use bullet points** - Easier to scan
- **Don't duplicate the diff** - Explain intent, not implementation
- Follow [CONTRIBUTING.md](/CONTRIBUTING.md) guidelines
- **For Python changes:** Run `ruff check --fix` and `ruff format` before finalizing

## Checklist

Before updating the PR:

- [ ] Verified existing description needs updating (not already complete)
- [ ] Summary accurately reflects the changes
- [ ] Breaking changes are clearly documented (if any)
- [ ] No unnecessary sections included
- [ ] Description is concise and scannable
