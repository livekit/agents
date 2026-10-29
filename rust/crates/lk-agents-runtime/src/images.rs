//! The declared image functions over this process's image worker and the host's frame reader.

use std::sync::Arc;

use lk_agents_abi::MediaRef;
use lk_agents_abi::media::images::{FrameReader, ImageEncodeOptions, ImageError, ImageInput};

use crate::process::{images, install_frame_reader, workers};

pub struct ImageFunctions;

impl lk_agents_abi::media::images::ImageFunctions for ImageFunctions {
    fn install_frame_reader(reader: Arc<dyn FrameReader>) {
        install_frame_reader(reader);
    }

    async fn encode_image(
        frame: MediaRef,
        format: String,
        quality: Option<u8>,
        resize_width: Option<u32>,
        resize_height: Option<u32>,
        resize_strategy: Option<String>,
    ) -> Result<Vec<u8>, ImageError> {
        let options =
            ImageEncodeOptions { format, quality, resize_width, resize_height, resize_strategy };
        let prepared = prepare(ImageInput::HostFrame(frame), options).await?;
        prepared.0.ok_or_else(|| ImageError("image encoder returned no bytes".into()))
    }

    async fn serialize_image_frame(
        frame: MediaRef,
        inference_width: Option<u32>,
        inference_height: Option<u32>,
    ) -> Result<(Vec<u8>, String), ImageError> {
        let resize = inference_width.zip(inference_height).filter(|(w, h)| *w > 0 && *h > 0);
        let options = ImageEncodeOptions {
            format: "JPEG".into(),
            quality: None,
            resize_width: resize.map(|(w, _)| w),
            resize_height: resize.map(|(_, h)| h),
            resize_strategy: resize.map(|_| "scale_aspect_fit".into()),
        };
        let prepared = prepare(ImageInput::HostFrame(frame), options).await?;
        let bytes = prepared.0.ok_or_else(|| ImageError("image encoder returned no bytes".into()))?;
        Ok((bytes, "image/jpeg".into()))
    }

    async fn serialize_image_url(
        url: String,
        mime_type: Option<String>,
    ) -> Result<(Option<Vec<u8>>, Option<String>, Option<String>, Option<String>), ImageError> {
        prepare(ImageInput::Url { url, mime_type }, ImageEncodeOptions::default()).await
    }
}

async fn prepare(
    input: ImageInput,
    options: ImageEncodeOptions,
) -> Result<lk_agents_abi::media::images::PreparedImage, ImageError> {
    let workers = workers().map_err(|e| ImageError(e.to_string()))?;
    images().prepare(&workers, input, options).await.map_err(ImageError)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use lk_agents_abi::utils::video::VideoFrame;
    use lk_agents_media::images::Images;

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
            buffer_type: 0,
            data: vec![127; 1920 * 1080 * 4].into(),
        };
        let workers = workers().unwrap();
        let images = Images::default();
        let mut encoding = Box::pin(images.prepare(
            &workers,
            ImageInput::Frame(frame),
            ImageEncodeOptions { format: "JPEG".into(), ..Default::default() },
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
        assert!(prepared.0.unwrap().starts_with(&[0xff, 0xd8, 0xff]));
    }
}
