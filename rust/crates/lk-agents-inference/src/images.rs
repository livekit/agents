use std::sync::Arc;

use lk_agents_abi::InferenceDetail;
use lk_agents_abi::llm::provider_format::ImageResolver;
use lk_agents_abi::llm::provider_format::image::SerializedImage;
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
    ) -> Result<
        lk_agents_abi::llm::provider_format::SerializedImage,
        lk_agents_abi::llm::provider_format::FormatError,
    > {
        Err(lk_agents_abi::llm::provider_format::FormatError::Image {
            image_id: image.id.clone(),
            reason: "no image preparation in this build".into(),
        })
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
        inference_detail: InferenceDetail,
        inference_width: Option<u32>,
        inference_height: Option<u32>,
    ) -> Result<SerializedImage, ImageError> {
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
                Ok(SerializedImage::inline(bytes, mime_type, inference_detail))
            }
            _ => Err(ImageError("image encoder returned no bytes".into())),
        }
    }

    async fn serialize_image_url(
        _env: &ProcessEnv,
        url: String,
        inference_detail: InferenceDetail,
        mime_type: Option<String>,
    ) -> Result<SerializedImage, ImageError> {
        let prepared =
            prepare(ImageInput::Url { url, mime_type }, ImageEncodeOptions::default()).await?;
        Ok(SerializedImage {
            inference_detail,
            mime_type: prepared.mime_type,
            data_bytes: prepared.data,
            external_url: prepared.external_url,
        })
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
#[path = "images_tests.rs"]
mod tests;
