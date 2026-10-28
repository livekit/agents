//! Image conversion and encoding run only on admitted native image workers.

use lk_agents_abi::llm::provider_format::{ImageResolver, SerializedImage};
use lk_agents_abi::media::images::{ImageEncodeOptions, ImageInput, PreparedImage};
use lk_agents_abi::utils::video::VideoFrame;
use lk_agents_abi::{ImageContent, ImageSource};
use lk_agents_media::images::MAX_IMAGE_BYTES;
use lk_agents_media::workers::{MediaWorkerClass, MediaWorkers};

pub async fn prepare(
    workers: &MediaWorkers,
    input: ImageInput,
    options: ImageEncodeOptions,
) -> Result<PreparedImage, String> {
    workers
        .try_spawn(MediaWorkerClass::Image, move || {
            lk_agents_media::images::prepare(input, options, host_frame)
        })
        .map_err(|e| e.to_string())?
        .await
        .map_err(|e| e.to_string())?
}

fn host_frame(frame: lk_agents_abi::MediaRef) -> Result<VideoFrame, String> {
    #[cfg(feature = "python")]
    {
        use pyo3::IntoPyObjectExt;
        use pyo3::types::PyAnyMethods;
        pyo3::Python::attach(|py| {
            let object = frame.into_bound_py_any(py)?;
            let width: u32 = object.getattr("width")?.extract()?;
            let height: u32 = object.getattr("height")?.extract()?;
            lk_agents_media::images::validate_dimensions(width, height)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            let frame = if object.getattr("type")?.extract::<i32>()? == 0 {
                object
            } else {
                object.call_method1("convert", (0,))?
            };
            VideoFrame::from_host(&frame, Some(width as usize * height as usize * 4))
        })
        .map_err(|e: pyo3::PyErr| e.to_string())
    }
    #[cfg(not(feature = "python"))]
    {
        let _ = frame;
        Err("host frame conversion requires a host runtime".into())
    }
}

/// How a realtime model prepares its images: on the image worker it already runs on.
pub struct NativeImages;

impl ImageResolver for NativeImages {
    fn resolve(&self, image: &ImageContent) -> Result<SerializedImage, String> {
        let input = match &image.image {
            ImageSource::Url(url) => {
                if url.len() > MAX_IMAGE_BYTES {
                    return Err("realtime image URL exceeds 16 MiB".into());
                }
                ImageInput::Url { url: url.clone(), mime_type: image.mime_type.clone() }
            }
            ImageSource::Frame(frame) => ImageInput::HostFrame(frame.clone()),
        };
        let resize = image
            .inference_width
            .zip(image.inference_height)
            .filter(|(width, height)| *width > 0 && *height > 0);
        let (data_bytes, mime_type, external_url, overridden_mime) =
            lk_agents_media::images::prepare(
                input,
                ImageEncodeOptions {
                    format: "JPEG".into(),
                    resize_width: resize.map(|(width, _)| width),
                    resize_height: resize.map(|(_, height)| height),
                    resize_strategy: resize.map(|_| "scale_aspect_fit".into()),
                    ..Default::default()
                },
                host_frame,
            )
            .map_err(|cause| format!("image {}: {cause}", image.id))?;
        if let Some(header) = overridden_mime {
            log::warn!(target: "livekit.agents", declared_mime = image.mime_type.as_deref().unwrap_or_default(), header_mime = header.as_str(); "image MIME declaration overrides data URL header");
        }
        Ok(SerializedImage {
            data_bytes,
            mime_type,
            external_url,
            inference_detail: image.inference_detail,
        })
    }
}

lk_agents_abi::inventory::submit! {
    lk_agents_abi::media::images::ImagePreparation {
        prepare: |input, options| Box::pin(async move {
            let workers = crate::process::workers().map_err(|e| e.to_string())?;
            prepare(&workers, input, options).await
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

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
        let mut encoding = Box::pin(lk_agents_abi::media::images::prepare(
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
