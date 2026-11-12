# Sentence splitting fixtures

The sentence splitter (`src/tokenize/sentences/`) is BlingFire's sentence breaker
(github.com/microsoft/BlingFire at 18e9a19, MIT, see `LICENSE`) ported rule for rule, with the
changes speech needs marked `Voice:` in `rules.rs`. The word lists in `src/tokenize/sentences/data/`
come from the same upstream rules, hence its `LICENSE` here.

`corpus.jsonl` holds short windows of text (one or two sentences), each with the split the
splitter gives it (`sentences`) and, where known, the split a voice wants (`expect`). It was
chosen by mutation testing from a corpus of about 47,000 documents kept outside the repository
(real voice-agent replies, LLM replies in eight languages, edge cases, Tatoeba sentences): every
rule deleted or inverted, every character set narrowed and every list entry left out changes the
split of some window here, which `the_corpus_notices_every_mutant` enforces.

- `the_corpus_splits_as_recorded`: every window splits as recorded.
- `no_group_scores_below_its_f1_floor`: precision, recall and F1 of the windows with `expect`,
  per source and language, against `f1_floors.tsv`. Ids name the source: `issue-*` (livekit/agents
  issues), `review-*` (past regressions), `probe-*` (one per list entry), `bench-*` (written after
  pragmatic_segmenter, Moses, UAX #29 and SaT edge cases), `voice-*` (read-out phone numbers,
  cards, codes, dates), `tatoeba-*` (tatoeba.org, CC BY 2.0 FR; only the boundaries between
  sentences are scored).

After a change to the rules, review the diff of `corpus.jsonl` and `f1_floors.tsv` and record it:

```bash
cargo test -p lk-agents-core --lib record_sentence_fixtures -- --ignored
```
