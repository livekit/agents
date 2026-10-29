//! Native image admission isolates text formatting and refuses oversized pixel work before encoding.
#![cfg(feature = "media-images")]

use lk_agents_abi::media::images::{ImageEncodeOptions, ImageInput};
use lk_agents_abi::utils::video::VideoFrame;
use lk_agents_abi::{ChatMessage, Role};
use lk_agents_media::workers::MediaWorkerClass;

#[tokio::test]
async fn text_turns_can_be_formatted_while_all_image_workers_are_busy() {
    let workers = lk_agents_runtime::process::workers().unwrap();
    let mut image_permits = Vec::new();
    while let Ok(permit) = workers.try_admit(MediaWorkerClass::Image) {
        image_permits.push(permit);
    }
    assert!(!image_permits.is_empty());
    let (messages, _) = <lk_agents_runtime::inference::llm::algorithms::LlmAlgorithms as lk_agents_abi::llm::provider_format::host::LlmAlgorithms>::prepare_provider_format(
        &lk_agents_runtime::process::env(),
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
    let workers = lk_agents_media::workers::MediaWorkers::new(Default::default()).unwrap();
    let result = lk_agents_runtime::images::prepare_on(
        &workers,
        std::sync::Arc::new(lk_agents_media::images::Images::default()),
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

#[test]
fn realtime_default_image_preparation_accepts_admitted_inline_urls() {
    use lk_agents_abi::llm::provider_format::ImageResolver;
    let image = lk_agents_abi::ImageContent {
        id: "data_image".into(),
        image: lk_agents_abi::ImageSource::Url("data:image/jpeg;base64,AQID".into()),
        inference_width: None,
        inference_height: None,
        inference_detail: Default::default(),
        mime_type: None,
    };
    let prepared = lk_agents_media::images::Images::default()
        .resolve(&image)
        .expect("the native constructor's resolver must accept a prepared inline image");
    assert_eq!(prepared.data_bytes.as_deref(), Some(&[1, 2, 3][..]));
    assert_eq!(prepared.mime_type.as_deref(), Some("image/jpeg"));
    let mut external = image.clone();
    external.image = lk_agents_abi::ImageSource::Url("https://example.test/image.jpg".into());
    let item = lk_agents_abi::ChatItem::Message(lk_agents_abi::ChatMessage::new(
        "external",
        lk_agents_abi::Role::User,
        vec![lk_agents_abi::ChatContent::Image(external)],
        std::time::SystemTime::UNIX_EPOCH,
    ));
    assert!(
        lk_agents_runtime::inference::realtime::native::wire::to_wire_item(
            &item,
            &lk_agents_media::images::Images::default()
        )
        .is_err(),
        "this realtime protocol requires inline images"
    );
    let mut oversized = image;
    oversized.image =
        lk_agents_abi::ImageSource::Url("x".repeat(lk_agents_media::images::MAX_IMAGE_BYTES + 1));
    assert!(lk_agents_media::images::Images::default().resolve(&oversized).is_err());
}
