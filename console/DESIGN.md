# the console design system

Read this before changing anything visual. It says where the references live,
which rules are not negotiable, and which mistakes have already cost time. It
is memdata's design system (`~/livekit/memdata/design/design-system.md`),
adapted: the same rules, this app's panels.

## The references, and what each one decides

| File | Decides |
| --- | --- |
| `../docs/clients.md`, `../docs/protocol/a2a.md`, `../docs/threads.md` | The supported conversation, protocol, and authorization contracts. The admin inspector uses the existing AgentDB service. |
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
the panel grey on the left; the agent's
progress is a quiet line with a pulse glyph; a tool's question to the person
is the agent asking, labelled "asks"; under every turn one muted mono line at
the chrome size: the task id (a link that opens the task in the inspector),
a plain-language outcome. Protocol state, metadata, worker identities, and storage
records belong to the explicit Admin inspector.

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
  the line, Escape clears, Cmd+K starts a new conversation.
- **The panel** (`chrome/panel.tsx`): a 25px header in the tertiary grey
  carrying the window label and a meta run, the body under it. `ListRow` is
  Blender's list: a 20px row, the hover fill, the accent when selected.
  `Field` is a label and its value in the control well. `Json` folds the
  document behind a row under a quiet summary.
- **The Admin inspector** (`drawer.tsx`) opens explicitly below the chat.
  It keeps the existing table and panel components, and its height is resizable.
  Snapshot requests start only when the developer enables the inspector. The
  tabs expose lanes, tasks, journal, inbox, A2A reply replay, read-only snapshot
  SQL, and connection information.
- **The conversations** (`chips.tsx`) form a row below the chat. Select one to
  load its history, or use New conversation. Titles come from the first message.

## Verifying a visual change

1. **Screenshot it.** Every "it looks right" that was not screenshotted has
   been wrong at least once.
2. **Run both engines.** `pnpm test:browser` covers `chromium` and `webkit`
   with hermetic service fixtures; `pnpm test:e2e` uses the live stack. The app is used in a Safari-based browser and a
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
- Browser reply and replay streams use authenticated JSON-RPC POST fetches.
  Cancel and release the reader when the task finishes, the view switches, or
  the component unmounts. Resume with the opaque SSE cursor; never interpret it.
- A splitter drag spends nearly all of its time with the pointer over the
  chat, which scrolls and selects text. `[data-splitting]` on the room shields
  the panes for the length of the drag.
