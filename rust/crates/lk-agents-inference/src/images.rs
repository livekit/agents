//! The declared image functions: provider images prepared on the media pool, host frames read
//! through the frame reader the host installs.

use std::sync::{Arc, OnceLock};

use lk_agents_abi::MediaRef;
use lk_agents_abi::media::images::{ImageEncodeOptions, ImageError, ImageInput, PreparedImage};

use lk_agents_media::pool::{JobClass, MediaPool};

use lk_agents_abi::llm::provider_format::ImageResolver;
use lk_agents_abi::media::images::FrameReader;
use lk_agents_net::env::ProcessEnv;

pub struct ImageFunctions;

/// The host's frame reader, installed once as the host starts.
static FRAME_READER: OnceLock<Arc<dyn FrameReader>> = OnceLock::new();

impl lk_agents_abi::media::images::FrameReaderInstall for ImageFunctions {
    /// The first one installed stays.
    fn install_frame_reader(reader: Arc<dyn FrameReader>) {
        let _ = FRAME_READER.set(reader);
    }
}

/// How this process prepares a request's images: media's preparation with the host's frame
/// reader, or none in a build without image support.
pub fn resolver() -> Arc<dyn ImageResolver + Send + Sync> {
    #[cfg(feature = "images")]
    return Arc::new(lk_agents_media::images::Images::new(FRAME_READER.get().cloned()));
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
        _env: &ProcessEnv,
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
        let bytes =
            prepared.0.ok_or_else(|| ImageError("image encoder returned no bytes".into()))?;
        Ok((bytes, "image/jpeg".into()))
    }

    async fn serialize_image_url(
        _env: &ProcessEnv,
        url: String,
        mime_type: Option<String>,
    ) -> Result<(Option<Vec<u8>>, Option<String>, Option<String>, Option<String>), ImageError> {
        prepare(ImageInput::Url { url, mime_type }, ImageEncodeOptions::default()).await
    }
}

async fn prepare(
    input: ImageInput,
    options: ImageEncodeOptions,
) -> Result<PreparedImage, ImageError> {
    prepare_on(&lk_agents_media::pool::media_pool(), resolver(), input, options)
        .await
        .map_err(ImageError)
}

/// Prepares `input` on an admitted image worker, keeping the caller's loop serviceable.
pub async fn prepare_on(
    media_pool: &MediaPool,
    images: Arc<dyn ImageResolver + Send + Sync>,
    input: ImageInput,
    options: ImageEncodeOptions,
) -> Result<PreparedImage, String> {
    media_pool
        .try_spawn(JobClass::Image, move || images.prepare(input, options))
        .map_err(|e| e.to_string())?
        .await
        .map_err(|e| e.to_string())?
}

#[cfg(all(test, feature = "images"))]
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
        let media_pool = MediaPool::new(Default::default()).unwrap();
        let mut encoding = Box::pin(prepare_on(
            &media_pool,
            Arc::new(Images::default()),
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
