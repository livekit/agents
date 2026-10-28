//! Awaitable image preparation; the runtime owns admission, conversion and encoding.

use crate::MediaRef;
use crate::utils::spawner::BoxFuture;
use crate::utils::video::VideoFrame;

pub enum ImageInput {
    Frame(VideoFrame),
    HostFrame(MediaRef),
    Url { url: String, mime_type: Option<String> },
}

#[derive(Clone)]
pub struct ImageEncodeOptions {
    pub format: String,
    pub quality: Option<u8>,
    pub resize_width: Option<u32>,
    pub resize_height: Option<u32>,
    pub resize_strategy: Option<String>,
}

impl Default for ImageEncodeOptions {
    fn default() -> Self {
        Self {
            format: "JPEG".into(),
            quality: Some(75),
            resize_width: None,
            resize_height: None,
            resize_strategy: None,
        }
    }
}

pub type PreparedImage = (Option<Vec<u8>>, Option<String>, Option<String>, Option<String>);

pub struct ImagePreparation {
    pub prepare:
        fn(ImageInput, ImageEncodeOptions) -> BoxFuture<'static, Result<PreparedImage, String>>,
}

inventory::collect!(ImagePreparation);

pub async fn prepare(
    input: ImageInput,
    options: ImageEncodeOptions,
) -> Result<PreparedImage, String> {
    let mut providers = inventory::iter::<ImagePreparation>.into_iter();
    let provider =
        providers.next().ok_or("image preparation service unavailable: no runtime linked")?;
    if providers.next().is_some() {
        return Err("image preparation service unavailable: multiple runtimes linked".into());
    }
    (provider.prepare)(input, options).await
}

#[crate::host(export(kw_only = "format", defaults(format = "JPEG".to_owned())))]
#[allow(clippy::too_many_arguments)]
pub async fn encode_image(
    frame: MediaRef,
    format: String,
    quality: Option<u8>,
    resize_width: Option<u32>,
    resize_height: Option<u32>,
    resize_strategy: Option<String>,
) -> Result<Vec<u8>, String> {
    let prepared = prepare(
        ImageInput::HostFrame(frame),
        ImageEncodeOptions { format, quality, resize_width, resize_height, resize_strategy },
    )
    .await?;
    prepared.0.ok_or_else(|| "image encoder returned no bytes".into())
}

#[crate::host(export)]
pub async fn serialize_image_frame(
    frame: MediaRef,
    inference_width: Option<u32>,
    inference_height: Option<u32>,
) -> Result<(Vec<u8>, String), String> {
    let resize = inference_width.zip(inference_height).filter(|(w, h)| *w > 0 && *h > 0);
    let prepared = prepare(
        ImageInput::HostFrame(frame),
        ImageEncodeOptions {
            format: "JPEG".into(),
            quality: None,
            resize_width: resize.map(|(w, _)| w),
            resize_height: resize.map(|(_, h)| h),
            resize_strategy: resize.map(|_| "scale_aspect_fit".into()),
        },
    )
    .await?;
    Ok((prepared.0.ok_or("image encoder returned no bytes")?, "image/jpeg".into()))
}

#[crate::host(export)]
pub async fn serialize_image_url(
    url: String,
    #[host(required)] mime_type: Option<String>,
) -> Result<(Option<Vec<u8>>, Option<String>, Option<String>, Option<String>), String> {
    prepare(ImageInput::Url { url, mime_type }, ImageEncodeOptions::default()).await
}
