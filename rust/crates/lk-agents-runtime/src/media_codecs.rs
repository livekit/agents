//! Host codec implementations over media calculators and the process-owned worker pool.

use std::sync::{Arc, Mutex};

use lk_agents_abi::media::codecs::{AudioDecoder as AudioDecoderApi, CodecFailure, DecoderOutput};
use lk_agents_media::Decoder;

pub struct AudioDecoder {
    inner: Mutex<Decoder>,
}

impl AudioDecoderApi for AudioDecoder {
    fn new(
        mime: String,
        sample_rate: Option<u32>,
        num_channels: u16,
    ) -> Result<Self, CodecFailure> {
        let workers = crate::process::workers().map_err(|error| CodecFailure(error.to_string()))?;
        let decoder = Decoder::with_workers(&workers, &mime, sample_rate, num_channels)
            .map_err(|error| CodecFailure(error.to_string()))?;
        Ok(Self { inner: Mutex::new(decoder) })
    }

    fn output_format(&self) -> Option<(u32, u16)> {
        self.inner.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).output_format_pair()
    }

    fn push(&self, data: &[u8]) -> Result<Vec<u8>, CodecFailure> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push_bytes(data)
            .map_err(|error| CodecFailure(error.to_string()))
    }

    fn flush(&self) -> Result<Vec<u8>, CodecFailure> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .flush_bytes()
            .map_err(|error| CodecFailure(error.to_string()))
    }

    fn set_on_output(&self, on_output: Option<Arc<dyn DecoderOutput>>) {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .set_on_output(on_output.map(|callback| Arc::new(move || callback.ready()) as _));
    }

    async fn aclose(&self) -> Result<(), CodecFailure> {
        let worker = self.inner.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).close();
        if let Some(worker) = worker {
            worker.await.map_err(|error| CodecFailure(error.to_string()))?;
        }
        Ok(())
    }
}

pub use lk_agents_media::StreamBuffer;

#[cfg(feature = "python")]
pub use lk_agents_media::blingfire::MediaTokenization;
