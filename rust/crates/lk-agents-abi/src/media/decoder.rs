//! Runtime decoding consumed by background playback and buffered room input.

use std::time::Duration;

use crate::AudioFrame;
use crate::utils::spawner::BoxFuture;

/// One bounded, admitted native decoder. Closing observes worker settlement.
pub trait FileDecoder: Send {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<AudioFrame>, String>>;
    fn close(&mut self, timeout: Duration) -> BoxFuture<'_, Result<(), String>>;
}

/// File playback format and repetition requested by the mixer.
pub struct FileDecoderOptions {
    pub path: String,
    pub looping: bool,
    pub sample_rate: u32,
    pub num_channels: u16,
}

/// A finite encoded head, normalized before it enters the live processing chain.
pub struct ByteDecoderOptions {
    pub mime: String,
    pub data: Vec<u8>,
    /// Raw PCM input format; containers describe their own source format.
    pub pcm_format: Option<(u32, u16)>,
    pub sample_rate: u32,
    pub num_channels: u16,
    pub max_input_bytes: usize,
    pub max_duration: Duration,
}

/// The runtime supplies admitted file and finite-buffer decoding when linked into the application.
pub struct FileDecoderProvider {
    pub open: fn(FileDecoderOptions) -> Result<Box<dyn FileDecoder>, String>,
    pub open_bytes: fn(ByteDecoderOptions) -> Result<Box<dyn FileDecoder>, String>,
}

inventory::collect!(FileDecoderProvider);

fn provider() -> Result<&'static FileDecoderProvider, String> {
    let mut providers = inventory::iter::<FileDecoderProvider>.into_iter();
    let provider = providers.next().ok_or("file decoder service unavailable: no runtime linked")?;
    if providers.next().is_some() {
        return Err("file decoder service unavailable: multiple runtimes linked".into());
    }
    Ok(provider)
}

pub fn open(options: FileDecoderOptions) -> Result<Box<dyn FileDecoder>, String> {
    (provider()?.open)(options)
}

pub fn open_bytes(options: ByteDecoderOptions) -> Result<Box<dyn FileDecoder>, String> {
    (provider()?.open_bytes)(options)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unexpected_open(_: FileDecoderOptions) -> Result<Box<dyn FileDecoder>, String> {
        panic!("ambiguous providers must be refused before opening a file")
    }

    fn unexpected_bytes(_: ByteDecoderOptions) -> Result<Box<dyn FileDecoder>, String> {
        panic!("ambiguous providers must be refused before decoding bytes")
    }

    inventory::submit! { FileDecoderProvider { open: unexpected_open, open_bytes: unexpected_bytes } }
    inventory::submit! { FileDecoderProvider { open: unexpected_open, open_bytes: unexpected_bytes } }

    #[test]
    fn conflicting_runtime_providers_refuse_file_open_before_effects() {
        let result = open(FileDecoderOptions {
            path: "clip.ogg".into(),
            looping: false,
            sample_rate: 48_000,
            num_channels: 1,
        });
        assert!(
            matches!(result, Err(error) if error == "file decoder service unavailable: multiple runtimes linked")
        );
    }
}
