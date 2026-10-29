//! Runtime decoding consumed by background playback.

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

/// The runtime supplies admitted file decoding when linked into the application.
pub struct FileDecoderProvider {
    pub open: fn(FileDecoderOptions) -> Result<Box<dyn FileDecoder>, String>,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn unexpected_open(_: FileDecoderOptions) -> Result<Box<dyn FileDecoder>, String> {
        panic!("ambiguous providers must be refused before opening a file")
    }

    inventory::submit! { FileDecoderProvider { open: unexpected_open } }
    inventory::submit! { FileDecoderProvider { open: unexpected_open } }

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
