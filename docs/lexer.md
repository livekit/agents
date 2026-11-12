# Lexer

What happens to a model's reply between the LLM and the person: what the voice says, what the
transcript shows, what the history keeps. One lexer reads each reply once, as it streams; every
consumer reads the same result, so the voice, the transcript and the history never disagree on
what was markup.

| The model writes | The voice says | The transcript shows | The history keeps |
| --- | --- | --- | --- |
| prose | it | it | it |
| reasoning, template tokens, turns it invents for others | nothing | nothing | nothing |
| an `<expr>` marker or the voice's own tag | its delivery, in the voice's syntax | nothing (`expression` on the chunk) | nothing |
| markdown syntax (`**`, `#`, `- `, `` ` ``) | nothing | it, for the frontend to render | nothing |
| `~~struck text~~` | nothing | it | nothing |
| a fenced code block | nothing | it | its code |
| a link `[text](url)` | the text | the link | `text (url)` |
| an emoji | nothing | it | it |

## What the model writes that is not for the person

Models write things a person must never hear: their reasoning, their chat template's tokens, the
next turn of the conversation, a tool call they wrote as text instead of calling it. These are
taken out of every model's stream once, before the reply is spoken, shown or saved, with the
model's own syntax:

- **Reasoning**: `<think>...</think>` for every model; a model with other delimiters has its own
  (Gemma's `<|channel>thought ... <channel|>`). A template that opens the reasoning itself, so the
  reply starts inside it and only the close arrives, is declared per model.
- **Turns**: a model that runs past its own turn writes the next one (`<|im_end|>` then
  `<|im_start|>user ...`). Only the assistant's turn is said; role headers, other roles' turns and
  anything after the end token are not. gpt-oss's channels follow the same rule: only `final` is
  said, `analysis` and `commentary` are not.
- **Control tokens** of the model's template, wherever they appear.
- **Tool calls written as text** (`<tool_call>{...}</tool_call>`, `[TOOL_CALLS]`) for models whose
  template has them: hidden and never run, with a warning that names the model.

It applies to every provider, below any `llm_node` override: an override that calls
`session.default.llm_node` gets the model's text with its syntax taken out, and an override's own
output is its own. `agent.run()` and `TextSession` read the same way. An attempt that produced only
reasoning before failing has said nothing, so it is retried like one that produced nothing.

## Expressive speech

`expressive=True` (or an `ExpressiveOptions`) on the session or an agent teaches the model a small
markup for delivery, and the voice turns it into the active TTS's own syntax. It applies only to a
TTS with a markup dialect; the guide is one system message, replaced each turn while expressive is
on and removed when it is off.

| Marker | Means |
| --- | --- |
| `<expr type="expression" label="warm"/>` | how the following words are delivered |
| `<expr type="sound" label="laugh"/>` | a non-verbal sound (`speech_steering.nonverbal_sounds` chooses which) |
| `<expr type="break" label="500ms"/>` | a pause |
| `<expr type="prosody" label="slow">...</expr>` | pace, volume or stress over the wrapped words |
| `<expr type="spell">A7X9</expr>` | spell the wrapped characters |

Each voice lowers the markers to what it supports (Cartesia's `<emotion>`, `<speed>` and `<spell>`,
Inworld's and Fish Audio's bracket cues, xAI's wrapping tags, Gemini's inline sounds and its part
style) and drops what it does not, keeping the words. The voice's own native tags, written by the
model or by you in `say()` text, are markup too and reach that voice as its own; a voice with no
dialect is sent the words alone.

Lowering writes two things for each chunk: the text the provider is sent and the words it says.
The words are what its audio is aligned to, so a cue such as `[laugh]` is never shown as a spoken
word, and a bracketed phrase that is prose (`see [the docs]`) is never taken for a cue. A cue a
voice echoes in its own word timestamps matches no word the transcript holds, so the transcript
skips it.

Only these are markup: `<expr>`, which is reserved, and the active TTS's native tags. Anything
else stays as written: `a<b`, `<3`, `x > y`, your own `<b>bold</b>`. A marker the model botched
(a forgotten quote, a bare word in the tag) is dropped and the words around it kept. A marker cut
off when the stream ends, by the end of the reply or by an interruption, is never spoken or shown.

The delivery a marker asks for reaches the room as the transcript's `lk.expression` attribute
(`{"expression": "happy", "mood": "happy"}`), for an avatar or a frontend to show. The history never
holds markup: the voice records the plain words heard, and it never rewrites past messages,
whatever expressive mode later turns into.

## Markdown and emoji

The voice never says markdown syntax or emoji; the transcript keeps both as written.

- Markdown syntax is not spoken, struck text and fenced code are not spoken, inline code is spoken
  without its backticks, a link is spoken as its text. Headings, list items, quotes, rules and
  fences count only at the start of a line; emphasis needs its closer on the same line and is not
  read inside a word (`2*3*4`, `snake_case`) or with blanks inside its delimiters (`5 * 3`).
- Emoji are not spoken.

Code is literal: markup quoted inside backticks is not markup.

## The transcript

The transcript reaches the transcription output as `TranscriptChunk` records, in order, each
either words the voice says or text with no speech:

- `spoken=True`: text the voice says, paced against the audio as it plays;
- `spoken=False`: text the voice does not say (markdown syntax, an emoji, a code block, a link's
  URL), shown as soon as everything before it was heard, at no cost to the pacing;
- `expression`: the delivery the model asked for from that chunk on, on the chunk where its marker
  sat.

So `**very** important 🎉` shows `**` as `very` starts, `**` once it is said, and `🎉` after
`important`, never ahead of the voice. A `transcription_node` receives these records and may return
records or plain strings; a `TranscriptionOutput` reads them through `capture_transcript(chunk)`.
A plain string handed to the chain (by a node or a custom output) never went through the reply's
reading, so its markup, any provider's tags, is taken out where it enters and its delivery kept.

A voice that times its words (Cartesia, Inworld and ElevenLabs send their word timings, and
LiveKit Inference asks for them by default; a voice behind `StreamAdapter` is timed sentence by
sentence) paces the transcript by those timings, and the transcript keeps its own text: each timed word is matched to the next spoken word, silent text
shows at the timing of the word after it, and a word the voice rewrote keeps its place. A voice
that reads `$5` as "five dollars" times `$5` across both of its words; one that matches nothing
nearby is mapped by position after four words, so the transcript never drifts far.

An interruption cuts the transcript at the last word heard. The history keeps the plain text of
what was heard: no markdown, no markup, the code and links kept, struck text gone.

## Sentences for the voice

Every TTS is fed through the same stage: the session's voice, a `tts.stream()` or `synthesize()`
of your own, a plugin, a `StreamAdapter`, each member of a `FallbackAdapter`. The stage splits the
text into chunks, paces them, and lowers each chunk to the provider's syntax; the provider only
sends what it receives. See [Sentences and pacing](models.md#sentences-and-pacing).

- Chunks are sentences, batched so short ones are not spoken alone; a provider that buffers text
  itself is sent words. `tts.options.sentence_tokenizer` decides the sentences: the built-in
  `tokenize.SentenceTokenizer()` by default, configurable, or a subclass of your own.
- A chunk never ends inside a tag or inside a wrapping marker's span: it waits for the rest, at
  most 600 characters.
- No chunk passes the provider's request limit: a longer sentence is cut at its last clause mark
  or word boundary before it, never inside a tag (1000 characters for a provider that declares no
  limit), and pacing never joins chunks past it.
- `tts.options.pacing` holds later chunks back while audio is still playing, so each request
  carries more context and sounds more natural; the first goes at once.

## How long text is held

Reading markup needs to see what comes next, so a little text is held while it could still turn
out to be markup. The holds are bounded, so the voice never waits on the reader:

| Text | Held until |
| --- | --- |
| prose | never held |
| `<`, `*`, `_`, `~`, `` ` ``, `[`, `!` | the next character shows whether it starts something |
| a tag | its `>`, at most 256 bytes; a longer one is read as botched |
| inline markdown (emphasis, code, link, strike) | its closer, on the same line, at most 160 bytes |
| a line marker (heading, list, rule, fence) | its line shows which, at most 160 bytes |
| a wrapping marker with no closer | the sentence splitter waits at most 600 characters |
| a sentence that does not end | the provider's request limit, else 1000 characters |
| reasoning | hidden as it arrives; only a delimiter's first characters are held |

## Design

This section is for contributors.

### One reader

`lk_agents_core::lexer` is the only reader of model text. A `Lexer` takes deltas and emits typed
tokens: prose, a markup `Tag`, an emoji, a markdown delimiter, or hidden text (reasoning, template
syntax, a botched or cut tag), each with its source text and its context (inside struck text, a
fenced block or inline code). Its `Rules` say what it recognizes: the model's `ModelSyntax`, a tag
`Vocabulary` (`LiveKit` for `<expr>` only, `Names` for `<expr>` plus a voice's natives or your own,
`All` for any tag), markdown and emoji. Everything a rule does not recognize is prose.

Each consumer reads the tokens through a `View`: `Model` (reasoning gone, for the LLM stage),
`Synthesis` (what the TTS is sent, tags canonical for lowering), `Spoken`, `Display` (the
transcript), `Plain` (the history). `Units` turns tokens into the transcript's spoken and silent
units. A voice reply is read once per segment (`SegmentText` in `lk-agents-voice`); the TTS gets
its synthesis view, the transcript chain its chunks, the item its plain view. The model's syntax is
taken out by `ModelText`, wrapped around the provider in `llm_leg` and `AgentRun`. Each provider's
lowering is a function of the tokens (`tts/markup/lower.rs`); the sentence splitter keeps whole the
spans `lexer::tag_spans` reports.

### The recognizer contract

At each position a recognizer answers: a match (these tokens over these bytes), a literal run, or
"need more". It says "need more" only while more text could change its answer, and never past its
window. Chunking invariance follows: the tokens never depend on how the stream was split, which
is what makes a cut, an interruption or a slow provider safe. At the end of the stream every
recognizer settles: markup and reasoning still open are dropped, an unclosed markdown opener is
literal.

Precedence decides what is literal inside what: the model's syntax first, then code (fenced,
inline), then markup tags (atomic), then links, then emphasis and strike, then line markers (only
at a line start). An emphasis or link body is read like any other text, so markup and code inside
emphasis keep their meaning. A delimiter that a longer delimiter could still become waits:
`<|im_start|>` is not settled until it is clear it is not `<|im_start|>assistant`.

### Tests

- Spec tables per recognizer and per provider's lowering, with each intended change from the
  former regex filters marked `// was:`.
- Generated replies over every construct and every voice dialect, each with its ground truth for
  every view: checked whole, under random chunking and one character at a time, at every cut
  point (a prefix of the words, no fragment of markup or reasoning), with the held bytes bounded
  after every delta.
- `cargo llvm-cov` keeps `lexer/` at 100% of lines and regions, and `cargo mutants` scoped to
  `lexer/*.rs` leaves no surviving mutant.
- The DST runs the real executors with a voice speaking a dialect and an expressive fake model
  (`Setup::tts_markup`, `Setup::expressive_model`): `no_markup_reaches_a_sink` holds over every
  transcription output, `aligned_transcript_follows_tts` checks each piece starts at a timing the
  voice gave and each word at the word the voice said, `cut_holds_only_shown_words` checks the
  history against what was shown.
