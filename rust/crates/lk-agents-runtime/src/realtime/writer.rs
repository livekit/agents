//! One independently polled socket writer. Framing and media conversion use admitted workers;
//! a stalled peer cannot suspend the owner's timer or command dispatch.

use super::wire::WireShape;
use lk_agents_abi::errors::{ApiError, TimeoutPhase};
use lk_agents_abi::llm::provider_format::ImageResolver;
use lk_agents_abi::{ApiConnectOptions, AudioByteStream, Clock, Elapsed, Ts};
use lk_agents_inference::realtime::dialect::{NUM_CHANNELS, SAMPLE_RATE};
use lk_agents_inference::realtime::machine::Outgoing;
use lk_agents_media::{FrameResampler, OnChange, Quality};
use lk_agents_net::transport::WsSender;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Batches retain bounded control traffic while one previous batch is in I/O.
pub(super) const WRITE_BACKLOG: usize = 32;

pub(super) const RESPONSE_RECEIPTS: usize = 32;

pub(super) struct ResponseSent {
    pub(super) generation: u64,
    pub(super) at: Ts,
    pub(super) req_id: String,
    pub(super) is_say: bool,
}

pub(super) struct PreparedFrame {
    pub(super) text: String,
    pub(super) response: Option<(String, bool)>,
}

pub(super) struct Encoder {
    resampler: FrameResampler,
    bstream: AudioByteStream,
}

impl Encoder {
    pub(super) fn new() -> Self {
        Self {
            resampler: FrameResampler::new(SAMPLE_RATE, Quality::Medium, OnChange::Discard),
            bstream: AudioByteStream::new(SAMPLE_RATE, NUM_CHANNELS, None, false),
        }
    }

    pub(super) fn encode(
        &mut self,
        effects: Vec<Outgoing>,
        images: &dyn ImageResolver,
        shape: &WireShape,
    ) -> Result<Vec<PreparedFrame>, String> {
        let mut frames = Vec::new();
        for effect in effects {
            match effect {
                Outgoing::AudioAppend { frame } => {
                    for frame in self.resampler.push(&frame).map_err(|error| error.to_string())? {
                        for chunk in self.bstream.push(&frame.data) {
                            frames.push(PreparedFrame {
                                text: super::wire::input_audio_buffer_append(&chunk.data)
                                    .to_string(),
                                response: None,
                            });
                        }
                    }
                }
                Outgoing::AudioClear => {
                    self.bstream = AudioByteStream::new(SAMPLE_RATE, NUM_CHANNELS, None, false);
                    self.resampler =
                        FrameResampler::new(SAMPLE_RATE, Quality::Medium, OnChange::Discard);
                    frames.push(PreparedFrame {
                        text: super::wire::input_audio_buffer_clear().to_string(),
                        response: None,
                    });
                }
                other => {
                    let response = match &other {
                        Outgoing::ResponseCreate { req_id, .. } => Some((req_id.clone(), false)),
                        Outgoing::ForceMessage { req_id, .. } => Some((req_id.clone(), true)),
                        _ => None,
                    };
                    let mut frame = super::encode::encode(other, images)?;
                    shape.outbound(&mut frame);
                    frames.push(PreparedFrame { text: frame.to_string(), response });
                }
            }
        }
        Ok(frames)
    }
}

pub(super) async fn run(
    mut effects: mpsc::Receiver<Vec<Outgoing>>,
    sender: Arc<dyn WsSender>,
    shape: WireShape,
    conn: ApiConnectOptions,
    clock: Arc<dyn Clock>,
    images: Arc<dyn ImageResolver + Send + Sync>,
    generation: u64,
    receipts: mpsc::Sender<ResponseSent>,
) -> Result<(), ApiError> {
    let workers = crate::process::workers()
        .map_err(|error| ApiError::new(error.to_string()).with_retryable(false))?;
    let mut encoder = Encoder::new();
    while let Some(batch) = effects.recv().await {
        let worker_images = images.clone();
        let worker_shape = shape.clone();
        let class = if batch.iter().any(|effect| matches!(effect,
            Outgoing::ItemCreate { item: lk_agents_abi::ChatItem::Message(message), .. }
                if message.content.iter().any(|content| matches!(content, lk_agents_abi::ChatContent::Image(_))))) {
            lk_agents_media::workers::MediaWorkerClass::Image
        } else {
            lk_agents_media::workers::MediaWorkerClass::Decoder
        };
        let prepared = workers
            .try_spawn(class, move || {
                let frames = encoder.encode(batch, worker_images.as_ref(), &worker_shape);
                (encoder, frames)
            })
            .map_err(|error| ApiError::new(error.to_string()).with_retryable(false))?;
        let (next_encoder, frames) = prepared.await.map_err(|error| {
            ApiError::new(format!("realtime frame worker failed: {error}")).with_retryable(false)
        })?;
        encoder = next_encoder;
        let frames = frames.map_err(|error| ApiError::new(error).with_retryable(false))?;
        for frame in frames {
            let deadline = conn.write_deadline(clock.now());
            match lk_agents_abi::before(clock.as_ref(), deadline, sender.send_text(frame.text))
                .await
            {
                Ok(Ok(())) => {
                    if let Some((req_id, is_say)) = frame.response {
                        receipts
                            .send(ResponseSent { generation, at: clock.now(), req_id, is_say })
                            .await
                            .map_err(|_| {
                                ApiError::new("realtime write receipt consumer closed")
                                    .with_retryable(false)
                            })?;
                    }
                }
                Ok(Err(error)) => {
                    return Err(lk_agents_net::transport::connect_error("realtime writer", error));
                }
                Err(Elapsed) => return Err(ApiError::timed_out(TimeoutPhase::Write)),
            }
        }
    }
    Ok(())
}
