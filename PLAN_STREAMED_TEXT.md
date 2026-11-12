# Streamed model text: one lexer

Status: approved design, in progress (2026-11-11).

## Problem

Model text arrives as deltas and is read by several consumers, each of which used its own parser:
the think filter (inference), the markdown and emoji `text_transforms` (voice), the TTS markup
`normalize`/`convert` regexes, the sentence splitter's `xml.rs`, the transcript stripper and the
pacer. They disagreed on what a tag is, held text unboundedly (`a<b`, `5 * 3`), leaked fragments
when a stream was cut mid-tag, and stripped the developer's own text (`<emphasis>` under any TTS,
`strip_assistant_markup` rewriting stored history).

## End state

One incremental lexer in `lk-agents-core/src/lexer/` emits typed tokens; every consumer is a pure
projection of the token stream.

- **Rules** (Rust-side configuration only, no public API yet):
  - `model: Option<ModelSyntax>`: what the model writes that is not for the person, per model
    (inference table, plugins supply theirs): hidden spans (reasoning `<think>`, Gemma's
    `<|channel>thought`, text-form tool calls `<tool_call>`/`[TOOL_CALLS]`, which also log a warning
    and are never run), control tokens (per-model lists only, e.g. `<|im_end|>`), gpt-oss harmony
    channels (only `final` is said), `starts_open` for templates that prefill the opener. Stray
    closers drop. Runs in the default `llm_node` (every provider; overriding the node opts out) and
    in the text session's LLM call.
  - `tags: Option<Vocabulary>`: `LiveKit` (`<expr>` only), `Names` (`expr` + the active TTS's
    natives, the default, or developer names), `All` (any tag-shaped text).
  - `markdown: bool`, `emoji: bool`: recognized as tokens (the former `text_transforms`).
- **Tokens** carry their source text and kind: prose, markup tag, emoji, markdown delimiter
  (emphasis, strike, inline code, link text/url, heading, list item, quote, rule, fence), and
  hidden (reasoning, botched or cut markup). Prose carries its context (inside strike, fence, code).
- **Projections**, each per token:
  - LLM output (`View::Model`): the model's own syntax hidden.
  - Spoken (TTS): markdown delimiters gone, struck text and fenced code not spoken, inline code
    spoken without backticks, emoji per `filter_emoji`, tags lowered by the provider (a function of
    `Tag`, no regexes).
  - Display (transcript): markdown kept as written, tags removed and published (`lk.expression`),
    emoji kept.
  - History: plain: no markdown delimiters, no tags; code and emoji kept. Never mutated after the
    fact (`strip_assistant_markup` deleted).
  - Sentence splitter: tag spans from the tokens.
- **Synchronizer** (`lexer::Sync`, done as a pure core): the transcript in source order as units,
  each a `Word` (said, shown as said) or `Silent` (shown or a tag's delivery, no speech). No unit
  is tied to another. The pacer paces words (or maps the TTS's aligned words by index) and shows a
  silent unit as soon as everything before it was heard, at zero cost; a cut lands on a unit
  boundary, and history is the plain projection of the tokens heard. Remaining: the pacer on it.

## The recognizer contract

`recognize(input, line_state, at_end) -> NoMatch | Match(len, token) | NeedMore`:

- monotone: `NeedMore` only when more input could change the answer, so output is independent of
  chunking by construction;
- bounded: inline constructs resolve within the line and `INLINE_WINDOW` chars, tags within
  `MAX_TAG`; a fence opener holds only to its line end (fenced content is never spoken, so it is not
  held);
- end of stream: markup and reasoning drop; an unclosed markdown delimiter is literal.

Precedence (what is literal inside what): reasoning > code (fence, inline) > markup tag (atomic) >
link > emphasis/strike > line markers (only at line start).

## Tests ("perfect coverage", with evidence)

- spec tables per recognizer, positive, negative and boundary;
- a ground-truth generator over every construct plus LLM-mistake mutations; invariants: chunk
  invariance (random splits, and every split of small documents), every cut point (prefix, no
  leak), held bytes bounded after every push, idempotent projections, words preserved;
- `cargo llvm-cov` 100% lines/regions on `lexer/`, `cargo-mutants` with no surviving mutant;
- a differential run against the former filters listing each intended change;
- DST rules: no markup or reasoning reaches any sink, history equals the heard plain words,
  pacing is unchanged by zero-cost tokens; a real-model corpus captured under `--evals`.

## Slices

1. [x] Lexer core: tags (vocabulary), think spans; think filter moved onto it.
2. [x] Markdown and emoji recognizers, token kinds, projections, `ModelSyntax` (spans, controls,
       harmony); 44 tests incl. generated replies (every view vs ground truth, random and
       per-character chunking, every cut). Coverage 96-100% per file.
2b. [ ] Close the coverage gaps to 100%, `cargo-mutants` clean on `lexer/`; differential corpus.
3. [x] TTS on the lexer: the sentence splitter keeps whole the voice's vocabulary
       (`TokenBufferOptions.markup`, `xml.rs` on `tag_spans`, a wrapper holds at most 600 chars);
       `convert` lowers tokens per provider (`tts/markup/lower.rs`; a wrapper wraps only when its
       closer is in the sentence) and `normalize` canonicalizes settled tags. The regex parser
       (`expr.rs`, `strip.rs`) is deleted; transcript stripping and `split_*_markup` run on the
       lexer (union vocabulary until slice 4b passes the provider). `strip_assistant_markup` is
       deleted: history is never rewritten.
3b. [x] Remove the public `xml_aware` (Python tokenizers, ABI) -- mapped to `Vocabulary::All`
       until then. Taken by the tokenizer rewrite (session agents-v2-14, confirmed 2026-11-11),
       which is rewriting these exact constructors on `lk_agents_core::lexer`, no new scanner.
       Done there: `tokenize/xml.rs` deleted, its gluing reimplemented in
       `tokenize/sentences/chunks.rs` on the lexer; `TokenBuffer`'s markup option gone. A user
       `sentence_tokenizer` on `StreamAdapter` gets the same gluing (`speakable_alone`), and an
       unclosed wrapper holds at most `MAX_WRAP` (600) characters on every path. Verified. Coordinate with the approved tokenizer rewrite (project_tokenizer_rewrite): it
       replaces the splitter rules, this the markup layer around them; both touch
       `tokenize/{mod,token_buffer}.rs`.
4. [x] Model syntax applies once, above every provider: `ModelText` (inference) wraps the LLM in
       `llm_leg` (voice and the default `llm_node`) and `AgentRun`; an override's own output is
       its own. Parser/gateway `TagFilter` removed (`LlmChoiceParser()` takes no model); the
       gateway asks the lexer whether an attempt said anything, so reasoning-only attempts retry.
4b. [x] Voice lexes once per segment (`SegmentText`): the TTS gets the synthesis view (the
       `text_transforms` pipeline and `filters.rs` deleted), the transcript chain gets
       `TranscriptChunk`s with `spoken` and `expression` (public, approved), the item the plain
       view. The pacer paces spoken chunks with its word tokenizer and shows silent ones at zero
       cost; the room publishes `chunk.expression` as `lk.expression` and parses nothing it got
       from the voice. A host's plain string (`capture_text`) is read by `HostText` (every
       provider's tags) at whichever end of the chain it enters. History is the plain view of
       what was heard.
5. [x] Aligned transcripts: each timed word maps onto the transcript's next spoken word
       (`AlignedTranscript`): silent text shows at the stamp's start at no cost, a stamp that finds
       no word within `LOOKAHEAD` is held and the words the next matching stamp skips take its
       span ("$5" read "five dollars"), and past the lookahead stamps map by position. The DST
       rule `aligned_transcript_follows_tts` now checks every piece starts at a stamp the voice
       gave and each word at the stamp of the word the voice said.
6. [~] DST: `Setup.tts_markup` (the voice's dialect, drawn last in the search so prior draws keep)
       and `Setup.expressive_model` (the fake model opens each reply with `<think>` and a marker);
       rule `no_markup_reaches_a_sink`. Remaining as first planned: no new fakes, two or three rules over the existing ones with the lexer's
       ground truth as oracle -- no markup or model syntax reaches a sink at any cut timing,
       history equals the heard plain words, silent units cost no pacing. Then the real-model
       corpus under `--evals`.
