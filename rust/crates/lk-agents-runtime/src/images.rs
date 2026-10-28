//! Image preparation and encoding on admitted workers. The SDK conversion callback runs on the
//! same worker as the encoder; callers never convert pixels or wait for native work on a loop.

use std::io::Write;

use lk_agents_media::images::{EncodeOptions, MediaError, RgbaFrame, encode_into};

use crate::media_workers::{MediaWorkerClass, MediaWorkerError, MediaWorkers};

#[derive(Debug, Clone, Copy)]
pub struct ImageLimits {
    pub source_pixels: usize,
    pub encoded_pixels: usize,
    pub encoded_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ImageError {
    #[error("image {kind} exceeds the admitted limit ({actual} > {limit})")]
    Limit { kind: &'static str, actual: usize, limit: usize },
    #[error("image dimensions must be positive and fit the address space")]
    InvalidDimensions,
    #[error("image conversion changed the admitted dimensions")]
    ChangedDimensions,
    #[error("image conversion failed: {0}")]
    Conversion(String),
    #[error(transparent)]
    Worker(#[from] MediaWorkerError),
    #[error(transparent)]
    Encode(#[from] MediaError),
    #[error("image worker failed: {0}")]
    Join(String),
}

/// Packed RGBA returned by a worker-safe SDK conversion. The callback owns its source until here.
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

#[derive(Clone)]
pub struct ImageEncoder {
    workers: MediaWorkers,
    limits: ImageLimits,
}

impl ImageEncoder {
    pub fn new(workers: MediaWorkers, limits: ImageLimits) -> Result<Self, ImageError> {
        if limits.source_pixels == 0 || limits.encoded_pixels == 0 || limits.encoded_bytes == 0 {
            return Err(ImageError::InvalidDimensions);
        }
        Ok(Self { workers, limits })
    }

    /// Admission precedes conversion and allocation. Cancellation drops only the awaiter; the
    /// shared worker owner keeps the source and its capacity until native work actually exits.
    pub async fn encode(
        &self,
        width: u32,
        height: u32,
        options: EncodeOptions,
        prepare: impl FnOnce() -> Result<RgbaImage, ImageError> + Send + 'static,
    ) -> Result<Vec<u8>, ImageError> {
        let source_pixels = pixels(width, height)?;
        check_limit("source pixels", source_pixels, self.limits.source_pixels)?;
        let output_pixels = encoded_pixels(width, height, &options)?;
        check_limit("encoded pixels", output_pixels, self.limits.encoded_pixels)?;
        let limit = self.limits.encoded_bytes;
        self.workers
            .try_spawn(MediaWorkerClass::Image, move || {
                let image = prepare()?;
                if (image.width, image.height) != (width, height) {
                    return Err(ImageError::ChangedDimensions);
                }
                let mut output = EncodedBytes { bytes: Vec::new(), limit, exceeded: false };
                let result = encode_into(
                    RgbaFrame { width, height, data: &image.data },
                    &options,
                    &mut output,
                );
                if let Err(err) = result {
                    if output.exceeded {
                        return Err(ImageError::Limit {
                            kind: "encoded bytes",
                            actual: limit.saturating_add(1),
                            limit,
                        });
                    }
                    return Err(err.into());
                }
                Ok(output.bytes)
            })?
            .await
            .map_err(|err| ImageError::Join(err.to_string()))?
    }
}

fn pixels(width: u32, height: u32) -> Result<usize, ImageError> {
    if width == 0 || height == 0 {
        return Err(ImageError::InvalidDimensions);
    }
    (width as usize)
        .checked_mul(height as usize)
        .filter(|pixels| pixels.checked_mul(4).is_some())
        .ok_or(ImageError::InvalidDimensions)
}

fn check_limit(kind: &'static str, actual: usize, limit: usize) -> Result<(), ImageError> {
    if actual > limit {
        return Err(ImageError::Limit { kind, actual, limit });
    }
    Ok(())
}

// Cover modes can allocate an intermediate image larger than the target canvas.
fn encoded_pixels(width: u32, height: u32, options: &EncodeOptions) -> Result<usize, ImageError> {
    let Some(resize) = options.resize_options else { return pixels(width, height) };
    let target = pixels(resize.width, resize.height)?;
    let (rw, rh) = lk_agents_media::images::resized_dimensions(width, height, resize)?;
    Ok(target.max(pixels(rw, rh)?))
}

struct EncodedBytes {
    bytes: Vec<u8>,
    limit: usize,
    exceeded: bool,
}

impl Write for EncodedBytes {
    fn write(&mut self, input: &[u8]) -> std::io::Result<usize> {
        if input.len() > self.limit - self.bytes.len() {
            self.exceeded = true;
            return Err(std::io::Error::other("encoded image byte limit exceeded"));
        }
        let required = self.bytes.len() + input.len();
        if required > self.bytes.capacity() {
            let capacity = required.max(self.bytes.capacity().saturating_mul(2)).min(self.limit);
            self.bytes
                .try_reserve_exact(capacity - self.bytes.len())
                .map_err(std::io::Error::other)?;
        }
        self.bytes.extend_from_slice(input);
        Ok(input.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media_workers::MediaWorkerLimits;

    fn encoder() -> ImageEncoder {
        let workers = MediaWorkers::new(
            tokio::runtime::Handle::current(),
            MediaWorkerLimits { total: 1, decoders: 1, recordings: 1, images: 1 },
        )
        .unwrap();
        ImageEncoder::new(
            workers,
            ImageLimits { source_pixels: 1024, encoded_pixels: 1024, encoded_bytes: 4096 },
        )
        .unwrap()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn sdk_conversion_and_encoding_leave_the_loop_available() {
        let encoder = encoder();
        let loop_thread = std::thread::current().id();
        let (release, wait) = std::sync::mpsc::channel();
        let (started, ready) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            encoder
                .encode(16, 16, EncodeOptions::default(), move || {
                    assert_ne!(loop_thread, std::thread::current().id());
                    started.send(()).unwrap();
                    wait.recv().unwrap();
                    Ok(RgbaImage { width: 16, height: 16, data: vec![255; 16 * 16 * 4] })
                })
                .await
        });
        ready.await.unwrap();
        tokio::task::yield_now().await;
        release.send(()).unwrap();
        let bytes = task.await.unwrap().unwrap();
        assert_eq!(&bytes[..3], &[0xff, 0xd8, 0xff]);
    }

    #[tokio::test]
    async fn oversized_images_are_refused_before_sdk_conversion() {
        let result = encoder()
            .encode(4096, 4096, EncodeOptions::default(), || {
                panic!("conversion must not run for refused images")
            })
            .await;
        assert!(matches!(result, Err(ImageError::Limit { kind: "source pixels", .. })));
    }

    #[tokio::test]
    async fn cover_intermediate_pixels_are_admitted_before_sdk_conversion() {
        use lk_agents_media::images::{ResizeOptions, ResizeStrategy};
        let options = EncodeOptions {
            resize_options: Some(ResizeOptions {
                width: 32,
                height: 1,
                strategy: ResizeStrategy::CenterAspectCover,
            }),
            ..EncodeOptions::default()
        };
        let result = encoder()
            .encode(1, 32, options, || {
                panic!("the intermediate resize must be refused before conversion")
            })
            .await;
        assert!(matches!(result, Err(ImageError::Limit { kind: "encoded pixels", .. })));
    }

    #[tokio::test]
    async fn encoded_byte_overflow_is_a_failure_instead_of_a_truncated_image() {
        let mut encoder = encoder();
        encoder.limits.encoded_bytes = 10;
        let result = encoder
            .encode(16, 16, EncodeOptions::default(), || {
                Ok(RgbaImage { width: 16, height: 16, data: vec![255; 16 * 16 * 4] })
            })
            .await;
        assert!(matches!(result, Err(ImageError::Limit { kind: "encoded bytes", .. })));
    }
}
