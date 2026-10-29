//! The native receiver's declared output retains the SDK frame and public segment marker.

use _core::generated::records::avatar::{audio_receiver_frame_from_py, audio_receiver_frame_to_py};
use lk_agents_abi::AudioFrame;
use lk_agents_abi::utils::video::AudioReceiverFrame;
use pyo3::prelude::*;

#[test]
fn receiver_frames_keep_the_sdk_audio_and_public_segment_end_types() {
    Python::initialize();
    Python::attach(|py| {
        py.import("livekit.agents").unwrap();
        let frame = AudioFrame::new(vec![0; 640], 16000, 1, 320);
        let audio = audio_receiver_frame_to_py(py, AudioReceiverFrame::Audio(frame)).unwrap();
        let audio_class = py.import("livekit.rtc").unwrap().getattr("AudioFrame").unwrap();
        assert!(audio.get_type().is(&audio_class));
        assert_eq!(audio.getattr("samples_per_channel").unwrap().extract::<u32>().unwrap(), 320);

        assert_eq!(
            audio_receiver_frame_from_py(&audio).unwrap(),
            AudioReceiverFrame::Audio(AudioFrame::new(vec![0; 640], 16000, 1, 320))
        );
        assert!(audio_receiver_frame_from_py(py.None().bind(py)).is_err());
        let video = lk_agents_abi::utils::video::VideoFrame {
            width: 1,
            height: 1,
            buffer_type: 0,
            data: vec![0u8; 4].into(),
        }
        .into_pyobject(py)
        .unwrap();
        assert!(audio_receiver_frame_from_py(&video).is_err());

        let end = audio_receiver_frame_to_py(py, AudioReceiverFrame::SegmentEnd).unwrap();
        let marker =
            py.import("livekit.agents.voice.avatar").unwrap().getattr("AudioSegmentEnd").unwrap();
        assert!(end.get_type().is(&marker));
        assert!(matches!(
            audio_receiver_frame_from_py(&end).unwrap(),
            AudioReceiverFrame::SegmentEnd
        ));

        let alias = py
            .import("livekit.agents._generated.avatar")
            .unwrap()
            .getattr("AudioReceiverFrame")
            .unwrap();
        let members = py.import("typing").unwrap().call_method1("get_args", (alias,)).unwrap();
        assert!(members.get_item(0).unwrap().is(&audio_class));
        assert!(members.get_item(1).unwrap().is(&marker));
    });
}
