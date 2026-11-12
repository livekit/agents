use self::markup::drop_bracket_cues;
use lk_agents_abi::TranscriptChunk;
use std::collections::BTreeMap;

pub mod markup;

/// Removes expressive cues from alignment batches and attaches their words to audio frames.
pub struct AudioTranscriptNormalizer {
    expressive: bool,
    segments: BTreeMap<String, (Vec<TranscriptChunk>, Vec<TranscriptChunk>)>,
}

impl AudioTranscriptNormalizer {
    pub fn new(expressive: bool) -> Self {
        Self { expressive, segments: BTreeMap::new() }
    }

    pub fn push_alignment(&mut self, segment_id: String, tokens: Vec<TranscriptChunk>) {
        let (pending, held) = self.segments.entry(segment_id).or_default();
        pending.extend(if self.expressive {
            drop_bracket_cues(tokens, held, false)
        } else {
            tokens
        });
    }

    pub fn take_audio_transcript(
        &mut self,
        segment_id: &str,
        is_final: bool,
    ) -> Vec<TranscriptChunk> {
        if is_final {
            let (mut pending, mut held) = self.segments.remove(segment_id).unwrap_or_default();
            pending.extend(drop_bracket_cues(Vec::new(), &mut held, true));
            pending
        } else {
            let (pending, _) = self.segments.entry(segment_id.to_owned()).or_default();
            std::mem::take(pending)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(text: &str, start: f64) -> TranscriptChunk {
        TranscriptChunk {
            text: text.into(),
            start_time: Some(start),
            end_time: Some(start + 0.1),
            ..Default::default()
        }
    }

    /// Transcript markup stripping keeps literal brackets and leaves no double space at a cue.
    /// https://github.com/livekit/agents/pull/6590 https://github.com/livekit/agents/pull/6717
    #[test]
    fn markup_stripping_keeps_brackets_and_cues_do_not_double_spaces() {
        for (parts, expected) in [
            (["Index [", "12] stays  literal"], "Index [12] stays  literal"),
            (["Hello <expr type=\"sound\" label=\"laugh\"/>", " world"], "Hello world"),
        ] {
            let mut stripper = markup::TranscriptMarkupStripper::new();
            let mut text = parts.iter().map(|part| stripper.push(part)).collect::<String>();
            text.push_str(&stripper.flush());
            assert_eq!(text, expected);
        }
    }

    /// Aligned transcripts keep brackets literal unless expressive, where a cue leaves no gap.
    /// https://github.com/livekit/agents/pull/6590 https://github.com/livekit/agents/pull/6717
    #[test]
    fn aligned_brackets_are_literal_without_expressive_and_cues_do_not_double_spaces() {
        for (expressive, parts, expected) in [
            (false, ["Index [", "12] stays  literal"], "Index [12] stays  literal"),
            (true, ["Hello [lau", "gh] world"], "Hello world"),
            (true, ["Hello [laugh]", " world"], "Hello world"),
        ] {
            let mut normalizer = AudioTranscriptNormalizer::new(expressive);
            let mut out = Vec::new();
            for (i, part) in parts.iter().enumerate() {
                normalizer.push_alignment("s".into(), vec![token(part, i as f64)]);
                out.extend(normalizer.take_audio_transcript("s", false));
            }
            out.extend(normalizer.take_audio_transcript("s", true));
            assert_eq!(out.iter().map(|t| t.text.as_str()).collect::<String>(), expected);
        }
    }

    #[test]
    fn interleaved_segments_close_only_their_own_partial_cues_and_keep_word_timing() {
        let mut normalizer = AudioTranscriptNormalizer::new(true);
        normalizer.push_alignment("first".into(), vec![token("[lau", 0.0)]);
        assert!(normalizer.take_audio_transcript("first", false).is_empty());

        normalizer.push_alignment("second".into(), vec![token("[unfinished", 1.0)]);
        assert!(normalizer.take_audio_transcript("second", false).is_empty());

        let second_end = normalizer.take_audio_transcript("second", true);
        assert_eq!(second_end[0].text, "[unfinished");
        assert_eq!(second_end[0].start_time, Some(1.0));

        normalizer.push_alignment("first".into(), vec![token("ghs] hello", 2.0)]);
        let first_end = normalizer.take_audio_transcript("first", true);
        assert_eq!(first_end.len(), 1);
        assert_eq!(first_end[0].text, "hello");
        assert_eq!(first_end[0].start_time, Some(2.0));
        assert_eq!(first_end[0].end_time, Some(2.1));
    }

    #[test]
    fn unclosed_cue_length_limit_is_evaluated_on_each_alignment_batch() {
        let mut normalizer = AudioTranscriptNormalizer::new(true);
        normalizer.push_alignment("segment".into(), vec![token("[", 0.0)]);
        let long = "a".repeat(260);
        normalizer.push_alignment("segment".into(), vec![token(&long, 1.0)]);
        normalizer.push_alignment("segment".into(), vec![token("] hello", 2.0)]);
        let transcript = normalizer.take_audio_transcript("segment", true);
        let text: String = transcript.iter().map(|token| token.text.as_str()).collect();
        assert_eq!(text, format!("[{long}] hello"));
    }
}
