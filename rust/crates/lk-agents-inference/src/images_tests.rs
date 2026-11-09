use std::time::Duration;

use lk_agents_abi::utils::video::{VideoBufferType, VideoFrame};
use lk_agents_abi::{ChatMessage, Role};
use lk_agents_media::pool::JobClass;

use super::*;

#[cfg(unix)]
fn thread_cpu() -> Duration {
    let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: the OS writes the calling thread's CPU time to this valid timespec.
    assert_eq!(unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) }, 0);
    Duration::new(time.tv_sec.try_into().unwrap(), time.tv_nsec.try_into().unwrap())
}

#[cfg(unix)]
#[tokio::test(flavor = "current_thread")]
async fn image_preparation_services_the_callers_loop_during_encoding() {
    let frame = VideoFrame {
        width: 1920,
        height: 1080,
        buffer_type: VideoBufferType::Rgba,
        data: vec![127; 1920 * 1080 * 4].into(),
    };
    let media_pool = MediaPool::new(Default::default()).unwrap();
    let mut encoding = Box::pin(prepare_on(
        &media_pool,
        ImageInput::Frame(frame),
        ImageEncodeOptions { format: ImageFormat::Jpeg, ..Default::default() },
    ));
    use std::future::{Future, poll_fn};
    let encoding = poll_fn(|cx| {
        let started = thread_cpu();
        let result = encoding.as_mut().poll(cx);
        assert!(thread_cpu().saturating_sub(started) <= Duration::from_millis(2));
        result
    });
    tokio::pin!(encoding);
    let mut observations = 0;
    let prepared = loop {
        tokio::select! {
            output = &mut encoding => break output.unwrap(),
            _ = tokio::time::sleep(Duration::from_millis(1)) => observations += 1,
        }
    };
    assert!(observations > 0, "the encoder must yield control before completion");
    assert!(prepared.data.unwrap().starts_with(&[0xff, 0xd8, 0xff]));
}

#[tokio::test]
async fn text_turns_can_be_formatted_while_all_image_workers_are_busy() {
    let media_pool = lk_agents_media::pool::media_pool();
    let mut image_permits = Vec::new();
    while let Ok(permit) = media_pool.try_admit(JobClass::Image) {
        image_permits.push(permit);
    }
    assert!(!image_permits.is_empty());
    let (messages, _) = <crate::llm::algorithms::LlmAlgorithms as lk_agents_abi::llm::provider_format::host::LlmAlgorithms>::prepare_provider_format(
        &ProcessEnv::current_runtime(),
        lk_agents_abi::llm::provider_format::ProviderFormat::OpenAi,
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
    let media_pool = lk_agents_media::pool::MediaPool::new(Default::default()).unwrap();
    let result = prepare_on(
        &media_pool,
        ImageInput::Frame(VideoFrame {
            width: 4097,
            height: 4097,
            buffer_type: VideoBufferType::I420,
            data: bytes::Bytes::new(),
        }),
        ImageEncodeOptions::default(),
    )
    .await;
    assert!(matches!(result, Err(error) if error.0.contains("16777216 pixels")));
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
    let prepared = lk_agents_media::images::Images
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
        crate::realtime::openai::wire::to_wire_item(&item, &lk_agents_media::images::Images)
            .is_err(),
        "this realtime protocol requires inline images"
    );
    let mut oversized = image;
    oversized.image =
        lk_agents_abi::ImageSource::Url("x".repeat(lk_agents_media::images::MAX_IMAGE_BYTES + 1));
    assert!(lk_agents_media::images::Images.resolve(&oversized).is_err());
}
