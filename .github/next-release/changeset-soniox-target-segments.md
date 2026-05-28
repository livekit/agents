---
"livekit-agents": patch
"livekit-plugins-soniox": patch
---

Surface per-run language segments end-to-end in the Soniox STT plugin, fixing two symmetric symptoms of the same underlying bug where `_TokenAccumulator._lang_segments` was computed but discarded in `send_endpoint_transcript` (and in the interim path). Both fixes live in the same block.

- Add `target_languages` and `target_texts` to `stt.SpeechData`, mirroring the existing `source_languages` / `source_texts` source-side fields.
- In translation mode, the Soniox plugin now populates `target_*` on `FINAL_TRANSCRIPT` and `INTERIM_TRANSCRIPT` / `PREFLIGHT_TRANSCRIPT` events, exposing per-run target-language breakdowns for code-switched two-way translation (e.g. `target_languages=["en", "es"]` / `target_texts=["Hello, how are you?", " Estoy bien, gracias."]` for the translation of `"Hello, ¿cómo estás? I'm doing fine, gracias."`).
- In non-translation mode, the plugin now populates `source_languages` / `source_texts` from the same accumulator (previously these were left `None`, making code-switched dictation look monolingual to consumers). For a Japanese-then-English utterance, consumers now see `source_languages=["ja", "en"]` instead of `None`.

`SpeechData.text` and `SpeechData.language` are unchanged for back-compat (still the full concatenation and the first translated/detected language, respectively). Fixes #5685.
