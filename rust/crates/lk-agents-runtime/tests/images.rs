//! Native image admission isolates text formatting and refuses oversized pixel work before encoding.
#![cfg(feature = "media-images")]

use lk_agents_abi::media::images::{ImageEncodeOptions, ImageInput};
use lk_agents_abi::utils::video::VideoFrame;
use lk_agents_abi::{ChatMessage, Role};
use lk_agents_runtime::media_workers::MediaWorkerClass;

#[tokio::test]
async fn text_turns_can_be_formatted_while_all_image_workers_are_busy() {
    let workers = lk_agents_runtime::process::workers().unwrap();
    let mut image_permits = Vec::new();
    while let Ok(permit) = workers.try_admit(MediaWorkerClass::Image) {
        image_permits.push(permit);
    }
    assert!(!image_permits.is_empty());
    let (messages, _) = <lk_agents_runtime::llm_algorithms::LlmAlgorithms as lk_agents_abi::llm::provider_format::host::LlmAlgorithms>::prepare_provider_format(
        "openai".into(),
        vec![ChatMessage::text("user", Role::User, "hello", std::time::UNIX_EPOCH).into()],
        None,
        true,
        false,
        None,
    )
    .await
    .expect("text turns must not compete with long image encodes");
    assert_eq!(messages[0]["content"], "hello");
}

#[tokio::test]
async fn oversized_pixels_are_refused_before_buffer_conversion_or_encoding() {
    let workers = lk_agents_runtime::media_workers::MediaWorkers::new(
        tokio::runtime::Handle::current(),
        Default::default(),
    )
    .unwrap();
    let result = lk_agents_runtime::images::prepare(
        &workers,
        ImageInput::Frame(VideoFrame {
            width: 4097,
            height: 4097,
            buffer_type: -1,
            data: bytes::Bytes::new(),
        }),
        ImageEncodeOptions::default(),
    )
    .await;
    assert!(matches!(result, Err(error) if error.contains("16777216 pixels")));
}
