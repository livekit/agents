#[cfg(feature = "media-ffmpeg")]
#[test]
fn compressed_heads_stop_at_their_decoded_duration_bound() {
    use lk_agents_voice::room_io::pre_connect_state::{HeadBytes, HeadError, MAX_DURATION};
    let seconds = MAX_DURATION.as_secs() as usize + 1;
    let mut encoder = lk_agents_media::ffmpeg::OggOpusEncoder::new(48_000, 1, 24_000).unwrap();
    let mut data = encoder.push(&vec![100; 48_000 * seconds]).unwrap();
    data.extend(encoder.finish().unwrap());
    let info = lk_agents_abi::voice::rtc::StreamInfo {
        mime_type: "audio/ogg;codecs=opus".into(),
        attributes: [("sampleRate".into(), "48000".into()), ("channels".into(), "1".into())]
            .into_iter()
            .collect(),
        ..Default::default()
    };
    let mut head = HeadBytes::new(&info).unwrap();
    head.push(&data).unwrap();
    assert_eq!(head.decode((24_000, 1)), Err(HeadError::Capacity("decoded duration")));
}
