//! One independently polled socket writer. Framing and media conversion use admitted workers;
//! a stalled peer cannot suspend the owner's timer or command dispatch.

use super::wire::WireShape;
use crate::realtime::dialect::{NUM_CHANNELS, SAMPLE_RATE};
use crate::realtime::machine::{Outgoing, ReplyKind};
use lk_agents_abi::errors::{ApiError, TimeoutPhase};
use lk_agents_abi::llm::provider_format::ImageResolver;
use lk_agents_abi::{ApiConnectOptions, Clock, Elapsed, Ts};
use lk_agents_media::AudioByteStream;
use lk_agents_media::{AudioResampler, OnChange, Quality};
use lk_agents_net::transport::WsSender;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Batches retain bounded control traffic while one previous batch is in I/O. Input is taken
/// only once this queue is empty, so audio waits in its own channel, under its policy.
pub(super) const WRITE_BACKLOG: usize = 32;

pub(super) const RESPONSE_RECEIPTS: usize = 32;

pub(super) struct ResponseSent {
    pub(super) generation: u64,
    pub(super) at: Ts,
    pub(super) req_id: String,
    pub(super) kind: ReplyKind,
}

pub(super) struct PreparedFrame {
    pub(super) text: String,
    pub(super) response: Option<(String, ReplyKind)>,
}

impl PreparedFrame {
    fn audio(text: String) -> Self {
        Self { text, response: None }
    }
}

pub(super) struct Encoder {
    resampler: AudioResampler,
    bstream: AudioByteStream,
}

impl Encoder {
    pub(super) fn new() -> Self {
        Self {
            resampler: AudioResampler::new(SAMPLE_RATE, Quality::Medium, OnChange::Discard),
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
                Outgoing::AudioAppend { frame } => self.append_audio(&frame, &mut frames)?,
                Outgoing::AudioClear => frames.push(self.clear_audio()),
                other => frames.push(encode_control(other, images, shape)?),
            }
        }
        Ok(frames)
    }

    /// Resamples `frame` to the wire rate and cuts it into append frames.
    fn append_audio(
        &mut self,
        frame: &lk_agents_abi::AudioFrame,
        frames: &mut Vec<PreparedFrame>,
    ) -> Result<(), String> {
        for frame in self.resampler.push_frame(frame).map_err(|error| error.to_string())? {
            frames.extend(self.bstream.push(&frame.data).iter().map(|chunk| {
                PreparedFrame::audio(
                    super::wire::input_audio_buffer_append(&chunk.data).to_string(),
                )
            }));
        }
        Ok(())
    }

    /// Drops the audio held back for framing and resampling, then clears the server's buffer.
    fn clear_audio(&mut self) -> PreparedFrame {
        *self = Encoder::new();
        PreparedFrame::audio(super::wire::input_audio_buffer_clear().to_string())
    }
}

/// A control frame; a response command carries its request id for the write receipt.
fn encode_control(
    effect: Outgoing,
    images: &dyn ImageResolver,
    shape: &WireShape,
) -> Result<PreparedFrame, String> {
    let response = match &effect {
        Outgoing::ResponseCreate { req_id, .. } => Some((req_id.clone(), ReplyKind::Generated)),
        Outgoing::ForceMessage { req_id, .. } => Some((req_id.clone(), ReplyKind::Say)),
        _ => None,
    };
    let mut frame = super::encode::encode(effect, images)?;
    shape.outbound(&mut frame);
    Ok(PreparedFrame { text: frame.to_string(), response })
}

/// One connection's socket writer: encodes each batch on the media pool and writes its frames
/// in order, receipting every response command once it is on the wire.
pub(super) struct Writer {
    pub(super) sender: Arc<dyn WsSender>,
    pub(super) shape: WireShape,
    pub(super) conn: ApiConnectOptions,
    pub(super) clock: Arc<dyn Clock>,
    pub(super) images: Arc<dyn ImageResolver + Send + Sync>,
    pub(super) generation: u64,
    pub(super) receipts: mpsc::Sender<ResponseSent>,
}

impl Writer {
    pub(super) async fn run(
        self,
        mut effects: mpsc::Receiver<Vec<Outgoing>>,
    ) -> Result<(), ApiError> {
        let mut encoder = Encoder::new();
        while let Some(batch) = effects.recv().await {
            let (next_encoder, frames) = self.prepare(encoder, batch).await?;
            encoder = next_encoder;
            for frame in frames {
                self.write(frame).await?;
            }
        }
        Ok(())
    }

    /// Encodes `batch` on the media pool; the encoder goes there and comes back with the frames.
    async fn prepare(
        &self,
        mut encoder: Encoder,
        batch: Vec<Outgoing>,
    ) -> Result<(Encoder, Vec<PreparedFrame>), ApiError> {
        let images = self.images.clone();
        let shape = self.shape.clone();
        let prepared = lk_agents_media::pool::media_pool()
            .try_spawn(batch_class(&batch), move || {
                let frames = encoder.encode(batch, images.as_ref(), &shape);
                (encoder, frames)
            })
            .map_err(|error| ApiError::new(error.to_string()).with_retryable(false))?;
        let (encoder, frames) = prepared.await.map_err(|error| {
            ApiError::new(format!("realtime frame worker failed: {error}")).with_retryable(false)
        })?;
        let frames = frames.map_err(|error| ApiError::new(error).with_retryable(false))?;
        Ok((encoder, frames))
    }

    /// Writes one frame within the write deadline, then receipts a response command.
    async fn write(&self, frame: PreparedFrame) -> Result<(), ApiError> {
        let deadline = self.conn.write_deadline(self.clock.now());
        match lk_agents_abi::before(
            self.clock.as_ref(),
            deadline,
            self.sender.send_text(frame.text),
        )
        .await
        {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                return Err(lk_agents_net::transport::connect_error("realtime writer", error));
            }
            Err(Elapsed) => return Err(ApiError::timed_out(TimeoutPhase::Write)),
        }
        let Some((req_id, kind)) = frame.response else { return Ok(()) };
        let sent = ResponseSent { generation: self.generation, at: self.clock.now(), req_id, kind };
        self.receipts.send(sent).await.map_err(|_| {
            ApiError::new("realtime write receipt consumer closed").with_retryable(false)
        })
    }
}

/// A batch that creates a message with an image is encoded as image work.
fn batch_class(batch: &[Outgoing]) -> lk_agents_media::pool::JobClass {
    let has_image = batch.iter().any(|effect| {
        matches!(effect,
            Outgoing::ItemCreate { item: lk_agents_abi::ChatItem::Message(message), .. }
                if message.content.iter().any(|content| matches!(content, lk_agents_abi::ChatContent::Image(_))))
    });
    if has_image {
        lk_agents_media::pool::JobClass::Image
    } else {
        lk_agents_media::pool::JobClass::Decoder
    }
}
