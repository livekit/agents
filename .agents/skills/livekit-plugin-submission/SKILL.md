---
name: livekit-plugin-submission
description: Fill out the LiveKit Plugin Partner Submission form for a company that built a LiveKit Agents plugin (STT, TTS, LLM, realtime model, avatar). Researches the answers, drafts the docs files, asks the partner only what it can't find, fills in the form and stops before Submit.
---

# LiveKit plugin submission helper

You're helping a partner company submit their LiveKit Agents plugin through LiveKit's Plugin Partner Submission form. The goal is to make this as easy as possible for them: find every answer you can yourself, ask them only what you can't, fill in the form for them and let them review it and click Submit.

Form link: https://livekit.notion.site/ca4b73e336fc47fc98debc0f10fbfc65?pvs=105 (public, no sign-in needed). If LiveKit sent the partner a different form link with this file, use that one instead.

If the partner attached or installed this file without saying much else, start Step 1 straight away.

## Ground rules

1. Never click Submit. Fill in the form, then hand it back to the partner to review and submit.
2. Only tick the two agreement checkboxes (package transfer and plugin maintenance) after the partner says yes to each one in this conversation. Explain both in plain words first.
3. Never invent people, email addresses, dates or legal names. If you can't confirm something from a source, ask.
4. Keep track of where each answer came from, so you can show the partner what to check.
5. LiveKit edits the docs and has the final say. Your docs drafts should be accurate and plain, with no marketing language.

## The flow

Overall, the partner gives you two or three links, answers one batch of questions, then reviews the filled form and submits it.

### Step 1: Ask for the starting links

In one short message, ask for:

- Company website
- PR link(s) on github.com/livekit/agents (Python) and/or github.com/livekit/agents-js (Node.js). If there's no PR yet, ask where the plugin code lives.
- Anything else they already have handy, such as API docs, a demo repo, an SVG logo or a brand kit link

If you're running inside a checkout of livekit/agents or livekit/agents-js, find the partner's plugin folder (in livekit/agents it's `livekit-plugins/livekit-plugins-<name>/`) and skip asking for the code.

### Step 2: Research (don't ask the partner anything here)

**Website and API docs**

- **Plugin name:** usually the company or product name.
- **Company legal name:** check the footer, terms of service or privacy policy. You'll confirm it with the partner in Step 3.
- **Website:** the homepage.
- **Service status:** Generally available, Beta or Preview, for the specific API the plugin uses.
- **API docs link:** the API reference the plugin is built on, not the docs homepage.
- **Supported models, languages and voices:** a link to their models or voices page, or a short list of model IDs and languages.
- **API key sign-up page:** the dashboard or sign-up page where developers create a key.
- **Social handles:** X, LinkedIn, YouTube and so on, from the site header or footer.
- **Brand assets:** a press kit or brand page, if one is public.
- **What makes your service different?:** a two- or three-sentence draft based on the partner's own docs. Use specific capabilities, not adjectives.

**Plugin types and OpenAI compatibility**

- **Plugin types:** STT, TTS, LLM, Realtime model, Virtual avatar or Other. Go by what the code implements (classes that subclass `stt.STT`, `tts.TTS`, `llm.LLM` or `llm.RealtimeModel`, or an avatar session). If there's no code yet, use the API docs. For Other, write a one-line description of the type.
- **OpenAI-compatible API?:** Yes, No or Partly. Check whether the API mirrors OpenAI endpoints such as `/v1/chat/completions`, `/v1/audio/speech`, `/v1/audio/transcriptions` or Realtime. If the answer is Yes, tell the partner in Step 3 that developers may be able to use LiveKit's existing OpenAI plugin with a custom `base_url`, so a separate plugin may not be needed. They can still submit.

**Package names**

- If you have a shell, run `pip index versions livekit-plugins-<name>` and `npm view @livekit/agents-plugin-<name> version`. "No matching distribution" from pip, or a 404 from npm, means the name is unclaimed.
- Without a shell, try `https://pypi.org/project/livekit-plugins-<name>/` and `https://www.npmjs.com/package/@livekit/agents-plugin-<name>`. PyPI often blocks automated fetches, so if neither check works, ask the partner in Step 3.
- If neither name exists, the answer is No. If a name exists, ask in Step 3 whether the partner published it. If someone else did, flag it.

**PR**

Use the `gh` CLI if it's available (`gh pr view <url> --comments` and `gh api repos/<owner>/<repo>/pulls/<n>/comments`). Otherwise read the PR page on github.com. To read the plugin code, open the files from the PR's head branch on `raw.githubusercontent.com/<fork owner>/agents/<branch>/<path>`.

- **SDKs:** Python for a livekit/agents PR, Node.js for a livekit/agents-js PR.
- **Maintainers:** collect GitHub handles from the PR author and commit authors. You'll need names and emails from the partner.
- **Review bot comments addressed?:** list the automated review comments (from Copilot, CodeRabbit, Devin and similar bots). Answer Yes only if every one is resolved or has a reply. Otherwise answer In progress, and give the partner a short list of the open ones.

**Demo or example app**

Check the partner's docs, their GitHub org (search for "livekit") and the PR description.

**Logo**

Only use an SVG the partner gives you. Check that it uses one color (a single fill or stroke, or `currentColor`), works on both light and dark backgrounds, and is under 100 KB. Never download and convert one from the web.

### Step 3: Ask the partner everything else in one message

Send one numbered list. Show what you already found, so they can confirm or correct it in the same reply. Include only the items you still need:

1. Company legal name: "I found <X>. Is that the entity that should own the terms?"
2. Primary contact: a business contact for status updates. A group address is best.
3. Maintainers: name and email for each GitHub handle you found, plus anyone else.
4. Security contact: a monitored address is best, for example security@theircompany.com.
5. Test access: how LiveKit gets free test keys or credits. LiveKit usually signs up with a livekit.io email, then the partner adds free credits and lets LiveKit invite teammates. Ask about their process and any limits.
6. Target launch date (optional): LiveKit needs at least two weeks from a complete submission.
7. Upcoming model launches (optional): model names, dates and whether they're under embargo.
8. Marketing contact (optional).
9. Co-marketing (optional): pick any of these: Blog post or press announcement, Social posts, Video featuring the LiveKit integration, Webinar featuring LiveKit, Integration or migration guide, Joint marketing video, Landing page featuring LiveKit, Credit promo for LiveKit developers, Joint campaign, Newsletter mention, Listing on your integrations or partners page.
10. Logo: a single-color SVG under 100 KB, if they haven't already sent one.
11. Package transfer agreement (yes or no). Explain it like this: after LiveKit approves the plugin and before it's merged, you move the package name, registry listing and repo to LiveKit's organization account. LiveKit owns and publishes it from then on, including if you stop maintaining it. LiveKit will send you the exact account. Never transfer it to anyone else.
12. Maintenance agreement (yes or no). Explain it like this: while the plugin is in LiveKit Agents, you keep it up to date with your service and with new Agents releases. That means adding new models, fixing bugs, responding to bug and security reports, and giving LiveKit a heads-up before breaking changes or before you stop maintaining it.

Also add any flags from Step 2: an OpenAI-compatible API, a package name someone else owns, open bot comments, or a target date less than two weeks away.

If they skip an optional item, leave that field blank. If they skip a required item, ask once more, briefly.

### Step 4: Draft the docs (one Markdown file per plugin type)

Name each file after its type: `stt.md`, `tts.md`, `llm.md`, `realtime.md` or `avatar.md`. Save them in a `livekit-submission/docs/` folder.

1. Find similar existing pages. Open the overview page for the type: `https://docs.livekit.io/agents/models/<type>.md`, where `<type>` is `stt`, `tts`, `llm`, `realtime` or `avatar`. It links to every provider page. Follow those links rather than guessing URLs, because the patterns differ by type (avatar pages, for example, are under `/avatar/plugins/`). Adding `.md` to any docs URL gives you the Markdown version. Read one or two providers with similar features.
2. Copy their structure, but skip any LiveKit Inference section. A typical page has:
   - Title and a one-line summary ("How to use <Name> <TYPE> with LiveKit Agents.")
   - Overview: one or two plain sentences.
   - Installation: `uv add "livekit-agents[<name>]~=<current minor>"` for Python and `pnpm add @livekit/agents-plugin-<name>` for Node.js, matching the version pattern on the reference page.
   - Authentication: the env var the plugin reads (from the code) and a link to the API key page.
   - Usage: a short `AgentSession` example that uses the real class name from the code, for each supported SDK.
   - Parameters: the main constructor arguments, with type, default and a one-line description, taken from the code. If you can't see the code, mark the section `TODO: confirm against plugin code`.
   - Feature sections, only for features the plugin actually supports (streaming, timestamps, custom voices, diarization, tool calling and so on).
   - Additional resources: the partner's docs, the Voice AI quickstart (`https://docs.livekit.io/agents/start/voice-ai.md`) and related LiveKit pages.
3. Put an HTML comment at the top of each file listing anything you weren't sure about.

### Step 5: Fill in the form

**If you have a browser tool** (Claude in Chrome, computer use or a Playwright-style tool), open the form and fill in every field you have an answer for. These notes come from a test run on the real form:

- The page loads fields as you scroll. Work top to bottom: fill what's on screen, scroll down, then look again. Element references from earlier go stale once you scroll.
- Text and URL fields: click the box and type.
- Radio buttons and checkboxes: find each option by its label text right before you click it. Never reuse screen coordinates from before a click, because picking a radio option adds a "Clear selection" link that pushes everything below it down. In testing, this made a second click land on the wrong answer.
- Target launch date: typing a date doesn't stick. Click the field, use the arrows to move to the right month, click the day, then click a blank area to close the picker. Check that the field shows the full date.
- File uploads (Draft docs and Logo): click the "Upload" button under the field. That adds a file input to the page, and you then attach the file to that input with your upload tool. If clicking Upload opened a file picker on the partner's screen, tell them to close it. If your tool can't upload files, tell the partner exactly which files to drag in.
- "If you picked Other, what type of plugin is it?" only appears after Other is ticked in Plugin types.
- The agreement checkboxes are labeled "We agree". Only tick them after the partner has said yes in this conversation.
- If the form asks for a sign-in, or a field won't take input, don't work around it. Give the partner the paste-ready sheet below for whatever is left. If the link won't open at all, tell them to ask their LiveKit contact for the current form link.

When you're done, scroll back to the top and check every answer on screen against your answer sheet before you hand the form back. Then stop. Don't click Submit.

**If you don't have a browser tool,** tell the partner in one line that Claude in Chrome would let you fill in the form for them next time. Then write `livekit-submission/answers.md` in the form's order, with each answer ready to copy, and give the partner the form link.

Either way, also save `livekit-submission/answers.md` so the partner has a record of what was entered.

Form order, with required fields marked:

1. Plugin name (required)
2. Company legal name (required)
3. Website (required)
4. Primary contact, name and email (required)
5. Plugin types (required)
6. If you picked Other, what type of plugin is it? (required if Other)
7. SDKs (required)
8. Is your API OpenAI-compatible?
9. Do you already own a livekit-plugins package name? (required)
10. Is your service generally available, in beta or in preview? (required)
11. PR link
12. Review bot comments addressed?
13. Maintainers, with name, email and GitHub handle for each (required)
14. Security contact, name and email (required)
15. We'll transfer the plugin package to LiveKit (required checkbox)
16. Plugin maintenance (required checkbox)
17. Test access setup (required)
18. What makes your service different?
19. Link to your API docs
20. Supported models, languages and voices
21. API key sign-up page
22. Demo or example app
23. Draft docs (Markdown)
24. Logo (SVG, black/white)
25. Brand assets
26. Target launch date
27. Upcoming model launches?
28. Marketing contact, name and email
29. Social handles
30. Co-marketing opportunities

### Step 6: Hand it back

Keep this to a few lines:

- The form is filled in and ready for their review. Or, if you couldn't fill it, here's the sheet to paste from.
- The two or three answers most worth double-checking, usually the legal name, the "What makes your service different?" draft and the docs files.
- Any files they still need to attach themselves.
- Any open flags (OpenAI compatibility, package name owner, open bot comments).
- When it looks right, they click Submit. LiveKit will follow up by email.
