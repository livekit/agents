//! Per-segment preparation of provider-aligned speech text.

use std::collections::BTreeMap;

use lk_agents_abi::TimedString;

use super::markup::drop_bracket_cues;

/// Removes expressive cues from alignment batches and attaches their words to audio frames.
pub struct AudioTranscriptNormalizer {
    expressive: bool,
    segments: BTreeMap<String, (Vec<TimedString>, Vec<TimedString>)>,
}

impl AudioTranscriptNormalizer {
    pub fn new(expressive: bool) -> Self {
        Self { expressive, segments: BTreeMap::new() }
    }

    pub fn push_alignment(&mut self, segment_id: String, tokens: Vec<TimedString>) {
        let (pending, held) = self.segments.entry(segment_id).or_default();
        pending.extend(if self.expressive {
            drop_bracket_cues(tokens, held, false)
        } else {
            tokens
        });
    }

    pub fn take_audio_transcript(&mut self, segment_id: &str, is_final: bool) -> Vec<TimedString> {
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

    fn token(text: &str, start: f64) -> TimedString {
        TimedString {
            text: text.into(),
            start_time: Some(start),
            end_time: Some(start + 0.1),
            ..Default::default()
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
