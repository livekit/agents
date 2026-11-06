//! The declared image functions: provider images prepared on the media pool, host frames read
//! through the frames themselves.

use std::sync::Arc;

use lk_agents_abi::llm::provider_format::ImageResolver;
use lk_agents_abi::media::images::{
    ImageEncodeOptions, ImageError, ImageFormat, ImageInput, PreparedImage, ResizeStrategy,
};
use lk_agents_abi::utils::video::RtcVideoFrame;
use lk_agents_media::pool::MediaPool;
use lk_agents_net::env::ProcessEnv;

pub struct ImageFunctions;

/// How this process prepares a request's images: media's preparation, or none in a build without
/// image support.
pub fn resolver() -> Arc<dyn ImageResolver + Send + Sync> {
    #[cfg(feature = "images")]
    return Arc::new(lk_agents_media::images::Images);
    #[cfg(not(feature = "images"))]
    Arc::new(NoImagePreparation)
}

/// No image preparation: every image is refused.
#[cfg(not(feature = "images"))]
struct NoImagePreparation;

#[cfg(not(feature = "images"))]
impl ImageResolver for NoImagePreparation {
    fn resolve(
        &self,
        image: &lk_agents_abi::ImageContent,
    ) -> Result<lk_agents_abi::llm::provider_format::SerializedImage, String> {
        Err(format!("image {}: no image preparation in this build", image.id))
    }
}

impl lk_agents_abi::media::images::ImageFunctions for ImageFunctions {
    type Env = ProcessEnv;

    async fn encode_image(
        _env: &ProcessEnv,
        frame: Arc<dyn RtcVideoFrame>,
        format: ImageFormat,
        quality: Option<u8>,
        resize_width: Option<u32>,
        resize_height: Option<u32>,
        resize_strategy: Option<ResizeStrategy>,
    ) -> Result<Vec<u8>, ImageError> {
        let options =
            ImageEncodeOptions { format, quality, resize_width, resize_height, resize_strategy };
        let prepared = prepare(ImageInput::HostFrame(frame), options).await?;
        prepared.data.ok_or_else(|| ImageError("image encoder returned no bytes".into()))
    }

    async fn serialize_image_frame(
        _env: &ProcessEnv,
        frame: Arc<dyn RtcVideoFrame>,
        inference_width: Option<u32>,
        inference_height: Option<u32>,
    ) -> Result<(Vec<u8>, String), ImageError> {
        let resize = inference_width.zip(inference_height).filter(|(w, h)| *w > 0 && *h > 0);
        let options = ImageEncodeOptions {
            format: ImageFormat::Jpeg,
            quality: None,
            resize_width: resize.map(|(w, _)| w),
            resize_height: resize.map(|(_, h)| h),
            resize_strategy: resize.map(|_| ResizeStrategy::ScaleAspectFit),
        };
        let prepared = prepare(ImageInput::HostFrame(frame), options).await?;
        match prepared {
            PreparedImage { data: Some(bytes), mime_type: Some(mime_type), .. } => {
                Ok((bytes, mime_type))
            }
            _ => Err(ImageError("image encoder returned no bytes".into())),
        }
    }

    async fn serialize_image_url(
        _env: &ProcessEnv,
        url: String,
        mime_type: Option<String>,
    ) -> Result<(Option<Vec<u8>>, Option<String>, Option<String>, Option<String>), ImageError> {
        let prepared =
            prepare(ImageInput::Url { url, mime_type }, ImageEncodeOptions::default()).await?;
        Ok((prepared.data, prepared.mime_type, prepared.external_url, prepared.overridden_mime))
    }
}

async fn prepare(
    input: ImageInput,
    options: ImageEncodeOptions,
) -> Result<PreparedImage, ImageError> {
    prepare_on(&lk_agents_media::pool::media_pool(), input, options).await
}

/// Prepares `input` on an admitted image worker, keeping the caller's loop serviceable.
#[cfg(feature = "images")]
pub async fn prepare_on(
    media_pool: &MediaPool,
    input: ImageInput,
    options: ImageEncodeOptions,
) -> Result<PreparedImage, ImageError> {
    use lk_agents_media::pool::JobClass;
    media_pool
        .try_spawn(JobClass::Image, move || lk_agents_media::images::prepare(input, options))
        .map_err(|e| ImageError(e.to_string()))?
        .await
        .map_err(|e| ImageError(e.to_string()))?
        .map_err(|e| ImageError(e.to_string()))
}

/// No image preparation in this build: every image is refused.
#[cfg(not(feature = "images"))]
pub async fn prepare_on(
    _media_pool: &MediaPool,
    _input: ImageInput,
    _options: ImageEncodeOptions,
) -> Result<PreparedImage, ImageError> {
    Err(ImageError("no image preparation in this build".into()))
}

#[cfg(all(test, feature = "images"))]
mod tests {
    use std::time::Duration;

    use lk_agents_abi::utils::video::{VideoBufferType, VideoFrame};

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
}
