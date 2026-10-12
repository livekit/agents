# the console design system

Read this before changing anything visual. It says where the references live,
which rules are not negotiable, and which mistakes have already cost time. It
is memdata's design system (`~/livekit/memdata/design/design-system.md`),
adapted: the same rules, this app's panels.

## The references, and what each one decides

| File | Decides |
| --- | --- |
| `../../agents-private/agent-db/docs/thread-registry-plan.md` 2 and 10.1 | The vocabulary and the product contract: session, lane, run, task, journal, inbox, stream; what demo 2 (the chat) and demo 3 (the debug view) show. When this and a screenshot disagree, this wins. |
| memdata's `design/exported-window.pdf` | The Blender Design System's own "Complete UI" export: the actual Blender window. The final word on how a control looks. Render a page with `pdftoppm -r 200 -png` and sample it rather than guessing. |
| `src/app/tokens.css` | The live values, memdata's Blender Dark set. **Source of truth for colour, size, radius, duration.** Do not restate its numbers elsewhere, here included: read the file. |

## The one rule that matters most

**Blender tells surfaces apart by fill, not by outline.** A panel is a
different grey from the room; a button is a different grey from the panel. It
does not draw a border to say "this is a thing". Reaching for `border:` is
almost always the wrong instinct here: change the fill instead. Hairlines exist
(`--line`) but they separate, they do not enclose.

Corollaries already settled: the inspector's splitter paints a 1px line, not
its whole grab area; the panes between the hairlines carry no frame; a message
is a fill on the room, not an outlined bubble.

## Colour

- Everything comes from `tokens.css`. `scripts/check-tokens.mjs` runs in
  `pnpm guard` and fails the build on a raw colour, font stack or duration
  anywhere else, and on an em dash in any file. It is not advisory.
- **The accent is a state signal, never decoration.** The selected session,
  the selected task, the pressed segment of a toggle, the primary action, the
  splitter while it is held, the focus ring. Not headings, not borders, not
  "some colour here".
- Status colours are for attention only. Silence is good. **There is no green**
  in status. `--failed-text` is for a task that ended `failed`, `canceled` or
  `rejected`, and for an error line; `--attention` is for a lapsed lease, an
  open run on a lane nobody leases, and a streams API that is not there. A
  completed task, a held lease and a drained inbox row are plain text.
- The inspector's icons are monochrome: with five icons in one column a
  category hue on each made a rainbow. The active icon takes the fill of the
  view it opens, so it reads as connected to it, as Blender's properties tabs
  do.

### A trap that has already bitten

A custom property that reads another one resolves **where it is declared**.
`--accent-hover: color-mix(in srgb, var(--accent) 82%, #fff)` on `:root`
resolves against `:root`'s accent, whatever an element below overrides. Use
`--accent-lift` (a `filter`) instead.

## Density and type

Blender's own metrics: 25px headers, 20px rows, 18px compact, 4px radius, 11px
Inter for chrome, 8px uppercase tracked for window headers, mono for ids and
data. The exact values are in `tokens.css`; use the `--h-*`, `--pad-*` and
`--fs-*` tokens (or their `h-row`, `p-3`, `text-micro` utilities) rather than
numbers.

The conversation is a measure (`--w-measure`), centred in whatever width the
page gives it: a chat column, not a sheet. Messages are content, not chrome,
so they get a reading size (`--fs-body`, `--lh-body`) and more air than
Blender's rows; everything around them keeps the chrome size. Messages are
fills on the room, the user's in the raised grey on the right, the agent's in
the panel grey on the left; tool calls are compact mono rows; the expert's
progress is a quiet line with a pulse glyph; a tool's question to the person
is the agent asking, labelled "asks"; under every turn one muted mono line at
the chrome size: the task id (a link that opens the task in the inspector),
the end state, the extension metadata, the worker and pid that served it.

Before the first message the stage is an invitation: the agent's one-line
description from its card as a title, one line under it, the composer right
under that, in the middle of the stage. From the first message on the
messages fill from the top and the composer sits at the foot. It is the same
element in both places (one grid, two row templates), so it keeps its focus.

**Less is better.** Nothing on the page says what the person did not ask
for: no status bar, no endpoint URL in the top bar (it is the agent name's
tooltip), no raw error text anywhere. A session that no longer exists is
dropped and the page starts a new conversation without a word; a registry
that does not answer is one quiet line in the sessions list; a turn that
failed is one plain sentence under that turn, in the failed colour, once.

## Icons

16px lucide glyphs at stroke 1.5, which is one CSS pixel at that size: the same
hairline the panel edges are drawn with (`src/components/chrome/icon.tsx`).
Blender's own icons are filled, and the weight of a solid shape says nothing
about how heavy an outline should be; heavier strokes read as bold next to
11px text and were tried and dropped in memdata. Every glyph fills its box.

## Components

- **One button, one field.** `.btn`, `.btnQuiet` and `.field` in `base.css`
  are the Blender control. Do not start a variant. The segmented toggle
  (`.segments`, `.segment`) is Blender's enum row: one well, the chosen
  segment in the accent. `.composer` is the field grown to hold more than
  one line: the same well, the ring on the well since the glyph is inside it.
- **The composer is a field and a glyph.** Enter sends, Shift+Enter breaks
  the line, Escape clears, Cmd+K starts a new conversation. The demo's knobs
  (policy, transport, the burst) are not in it: they are behind the quiet
  "demo" word in the top bar (`demo.tsx`, a native popover), and the composer
  shows them as one tiny label only when they are off their defaults.
- **The panel** (`chrome/panel.tsx`): a 25px header in the tertiary grey
  carrying the window label and a meta run, the body under it. `ListRow` is
  Blender's list: a 20px row, the hover fill, the accent when selected.
  `Field` is a label and its value in the control well. `Json` folds the
  document behind a row under a quiet summary.
- **The inspector** (`inspector.tsx`) is Blender's properties editor on the
  right of the chat: a vertical icon column against the view it switches,
  closed by default so the chat is the page, resized on its left edge,
  never wider than half the window. A page with no conversation opens it
  closed whatever was open last time; the Stream icon is disabled until a
  task is selected (selecting one, in Tasks or from a turn's task link,
  opens it). Every view with nothing to show says so in one line. On a
  narrow window the pinned sessions list yields to it and comes back when
  it closes.
- **The sessions rail** (`panels/sessions.tsx`) is the inspector's twin on
  the left: a strip with two icons. Hovering it peeks the list over the
  stage (nothing moves); clicking the icon pins it beside the stage. A row
  is the end of the id (the whole id on hover), the lane count only when
  there is more than one, and a relative time; the selected row takes the
  accent.

## The panel inventory

| Panel | Shows |
| --- | --- |
| Sessions | the subject's sessions newest first from the registry, peeking on hover and pinned on click: the end of the id, lanes when more than one, last active; New conversation; selecting one loads it everywhere; one quiet line when the registry does not answer |
| Conversation | the chat: the invitation before the first message, then messages, tool calls, progress, questions, the reply streaming, and under each turn the task (a link into the inspector), its end state and the worker and pid; the composer, a field and a glyph, with the off-default demo knobs as one tiny label |
| Demo | behind the top bar's "demo" word: the policy toggle (interrupt or inbox: `livekit/channel` a2a or sms), the transport toggle (HTTP+SSE or WebSocket), a burst of two messages |
| Lanes | the session first (id, tip, the worker that served the last turn), then per lane: version, lease holder, lease until (counting down on the server's clock), waiter, drained seq, state |
| Tasks | the task table: state, status message, attempts, due and deadline, the merge and the interrupt off the task's metadata; selecting one opens its stream |
| Journal | run_started and run_ended paired by run id, and every other event, newest last |
| Inbox | the rows received while a turn ran, pending or drained |
| Stream | the selected task's answer stream, replayed from the start and followed live; not a destination on its own, it opens when a task is selected |

## Verifying a visual change

1. **Screenshot it.** Every "it looks right" that was not screenshotted has
   been wrong at least once.
2. **Run both engines.** `pnpm test:e2e` covers `chromium` and `webkit`
   against the live stack. The app is used in a Safari-based browser and a
   whole class of breakage only appears there.
3. **Assert on `data-` attributes, not class substrings.** Every panel, pane,
   row and entry carries one (`data-panel`, `data-pane`, `data-entry`,
   `data-task`, `data-lane`, `data-lease-state`, `data-stream-state`).

### Things worth knowing before you fight them

- React runs a state updater twice in development. Numbering stream frames
  inside one gave every frame two indices; anything with a side effect or a
  counter happens before `setState`, never inside it.
- Resetting state in an effect when a key changes is a lint error here
  (`react-hooks/set-state-in-effect`) and a render wasted. State that
  belongs to a session or a task is stored keyed by it and derived.
- `EventSource` reconnects when the server closes the connection and replays
  the stream from the start. A `streamClosed` control frame closes the source
  on the spot, outside any updater.
- A splitter drag spends nearly all of its time with the pointer over the
  chat, which scrolls and selects text. `[data-splitting]` on the room shields
  the panes for the length of the drag.
