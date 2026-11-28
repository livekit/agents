use super::audio_byte_stream::AudioByteStream;
use crate::test_alloc::allocated;

/// Framing allocates only linear storage while preserving every PCM byte. https://github.com/livekit/agents/pull/4247
#[test]
#[ignore = "https://github.com/livekit/agents/pull/4247 reproduces: framing repeatedly allocates and copies the unconsumed tail"]
fn extracting_many_frames_does_not_reallocate_the_remaining_tail_each_time() {
    for frames in [128, 256, 512] {
        let input: Vec<_> = (0..frames * 320).map(|i| (i % 251) as u8).collect();
        let mut stream = AudioByteStream::new(16_000, 1, Some(160), false);
        let (out, allocated) = allocated(|| stream.push(&input));
        assert_eq!(out.len(), frames);
        assert_eq!(out.into_iter().flat_map(|f| f.data).collect::<Vec<_>>(), input);
        assert!(
            allocated < input.len() * 8,
            "framing {frames} frames allocated {allocated} bytes for {} input bytes",
            input.len()
        );
    }
}
