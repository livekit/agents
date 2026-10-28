use super::Tag;

#[test]
fn reserved_marker_values_are_unknown_audio_without_a_producer() {
    for chunk in 0..=Tag::MAX_CHUNK as i16 {
        for sample in [chunk, -chunk, Tag::MIC | chunk, -(Tag::MIC | chunk)] {
            assert_eq!(Tag::decode(sample), Tag::Unknown(sample), "sample {sample}");
        }
    }
    assert_eq!(Tag::decode(i16::MIN), Tag::Unknown(i16::MIN));
    // The first representable producer remains producer zero, including both marker sides.
    assert_eq!(Tag::decode(Tag::synthesized(0, 0)), Tag::Synthesized { tts: 0, chunk: 0 });
    assert_eq!(Tag::decode(Tag::generated(0, 0)), Tag::Generated { generation: 0, chunk: 0 });
    assert_eq!(Tag::decode(Tag::fed(0, 0)), Tag::Fed { stream: 0, chunk: 0 });
    assert_eq!(Tag::decode(Tag::mic(0, 0)), Tag::Mic { mic: 0, chunk: 0 });
}
