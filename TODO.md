# TODO

## Correct history from a late playback report

Rule: what is stored is what the person heard; the audio sink's playback report is the truth.

Today, when a speech is cut and the sink has not reported within `CUT_REPORT_WAIT` (1 s), the
interrupted message is committed with the words the transcript synchronizer showed by the clear
(`Completion::CutUnreported`, `lk-agents-voice/src/generation/machine/playout.rs`). A report that
arrives later (before the 5 s `UNREPORTED_CUT_GRACE`) is not used for history, so when it differs
the stored message can be wrong:

- the sink played further than the synchronizer had shown: heard words are missing;
- the sink played the whole segment (one that ignores `clear_buffer`): the message is stored as
  interrupted and truncated.

Rare (needs a sink that reports more than 1 s after a clear), so deferred. Agreed shape:

1. A late report corrects the committed item in place (same id) through the existing
   `replace_items`, which re-sends `conversation_item_added`.
2. The thread's saved copy is written once: an interrupted speech whose report has not arrived
   holds its save until the report or the 5 s bound, then saves the final text. The agent's next
   reply is not delayed.
3. The live transcript is not reopened: it stays as the person saw it; only history is corrected.
