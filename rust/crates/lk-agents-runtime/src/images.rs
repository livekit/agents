//! Image conversion and encoding run only on admitted native image workers.

use std::io::Write;

use lk_agents_abi::media::images::{ImageEncodeOptions, ImageInput, PreparedImage};
use lk_agents_abi::utils::video::VideoFrame;
use lk_agents_media::images::{
    EncodeOptions, ImageFormat, ResizeOptions, ResizeStrategy, RgbaFrame,
};

use lk_agents_media::workers::{MediaWorkerClass, MediaWorkers};

const MAX_PIXELS: u64 = 4096 * 4096;
pub(crate) const MAX_IMAGE_BYTES: usize = 16 * 1024 * 1024;

pub async fn prepare(
    workers: &MediaWorkers,
    input: ImageInput,
    options: ImageEncodeOptions,
) -> Result<PreparedImage, String> {
    workers
        .try_spawn(MediaWorkerClass::Image, move || prepare_on_worker(input, options))
        .map_err(|e| e.to_string())?
        .await
        .map_err(|e| e.to_string())?
}

/// The caller retains Image worker admission until this synchronous preparation settles.
pub(crate) fn prepare_on_worker(
    input: ImageInput,
    options: ImageEncodeOptions,
) -> Result<PreparedImage, String> {
    let frame = match input {
        ImageInput::Url { url, mime_type } => {
            if url.len() > MAX_IMAGE_BYTES {
                return Err("image URL exceeds 16 MiB".into());
            }
            return lk_agents_media::images::serialize_image_url(&url, mime_type.as_deref())
                .map_err(|e| e.to_string());
        }
        ImageInput::Frame(frame) => frame,
        ImageInput::HostFrame(frame) => host_frame(frame)?,
    };
    validate_dimensions(frame.width, frame.height)?;
    if frame.buffer_type != 0 {
        return Err("image encoder requires an RGBA frame".into());
    }
    let format = options.format.parse::<ImageFormat>().map_err(|e| e.to_string())?;
    let resize_options =
        match (options.resize_width, options.resize_height, options.resize_strategy) {
            (Some(width), Some(height), Some(strategy)) => {
                validate_dimensions(width, height)?;
                let strategy = strategy.parse::<ResizeStrategy>().map_err(|e| e.to_string())?;
                Some(ResizeOptions { width, height, strategy })
            }
            (None, None, None) => None,
            _ => {
                return Err(
                    "resize needs all three of width, height and strategy, or none of them".into(),
                );
            }
        };
    let frame = RgbaFrame { width: frame.width, height: frame.height, data: &frame.data };
    if let Some(resize) = resize_options {
        let (width, height) =
            lk_agents_media::images::resized_dimensions(frame.width, frame.height, resize)
                .map_err(|e| e.to_string())?;
        validate_dimensions(width, height)?;
    }
    let options = EncodeOptions { format, quality: options.quality, resize_options };
    let mut output = ImageBytes(Vec::new());
    lk_agents_media::images::encode_into(frame, &options, &mut output)
        .map_err(|e| e.to_string())?;
    let mime = match format {
        ImageFormat::Jpeg => "image/jpeg",
        ImageFormat::Png => "image/png",
    };
    Ok((Some(output.0), Some(mime.into()), None, None))
}

fn validate_dimensions(width: u32, height: u32) -> Result<(), String> {
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err("image dimensions must contain between 1 and 16777216 pixels".into());
    }
    Ok(())
}

struct ImageBytes(Vec<u8>);

impl Write for ImageBytes {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self.0.len().saturating_add(bytes.len()) > MAX_IMAGE_BYTES {
            return Err(std::io::Error::other("encoded image exceeds 16 MiB"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
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
            validate_dimensions(width, height).map_err(pyo3::exceptions::PyValueError::new_err)?;
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
