//! Track publications, local and remote, and the frame conversion a real `rtc.AudioStream` does.

use lk_agents_abi::voice::rtc::RoomEvent;
use lk_agents_abi::voice::rtc::RtcError;
use lk_agents_abi::voice::rtc::capabilities::TrackPublication;
use lk_agents_abi::voice::rtc::{
    AudioStreamOptions, AudioTrackFeature, TrackSid, TrackSource, VideoFrameRef,
};
type Result<T> = std::result::Result<T, RtcError>;
use super::FakeRemoteParticipant;
use lk_agents_abi::AudioFrame;
use lk_agents_abi::StreamSource;
use lk_agents_abi::voice::rtc::AudioInputStream;
use lk_agents_abi::voice::rtc::RoomEventSink;
use lk_agents_runtime::room_io::bounded_lossy::{BoundedLossy, LossyTx, Pushed};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use tokio::sync::Notify;

/// Local or remote depending on whether it has an `owner`.
pub struct FakeTrackPublication {
    pub(super) sid: TrackSid,
    pub(super) name: String,
    source: TrackSource,
    bus: Arc<dyn RoomEventSink>,
    owner: Option<Weak<FakeRemoteParticipant>>,
    subscribed: AtomicBool,
    subscription: Notify,
    state: Mutex<PublicationState>,
}

#[derive(Default)]
struct PublicationState {
    features: Vec<AudioTrackFeature>,
    /// Dropping them all ends the streams.
    streams: Vec<OpenStream>,
    video_streams: Vec<LossyTx<VideoFrameRef>>,
    /// Kept past `close_streams`, so a test can inspect it after the track went away.
    requested: StreamRequests,
}

/// The options are kept because a real `rtc.AudioStream` converts to the requested rate and
/// channel count, so a fake that ignored them could not fail a wrong-rate test.
struct OpenStream {
    tx: LossyTx<AudioFrame>,
    sample_rate: Option<u32>,
    num_channels: Option<u16>,
}

/// The options a real `rtc.AudioStream` acts on are host-side, so recording them is the only way
/// to assert they were forwarded.
#[derive(Clone, Default)]
pub struct StreamRequests(Vec<AudioStreamOptions>);

impl StreamRequests {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn get(&self, n: usize) -> Option<&AudioStreamOptions> {
        self.0.get(n)
    }
}

impl FakeTrackPublication {
    pub(super) fn new(
        sid: TrackSid,
        source: TrackSource,
        bus: Arc<dyn RoomEventSink>,
        owner: Option<Weak<FakeRemoteParticipant>>,
        subscribed: bool,
    ) -> Arc<Self> {
        Arc::new(FakeTrackPublication {
            sid,
            name: String::new(),
            source,
            bus,
            owner,
            subscribed: AtomicBool::new(subscribed),
            subscription: Notify::new(),
            state: Mutex::new(PublicationState::default()),
        })
    }

    pub fn set_audio_features(&self, features: Vec<AudioTrackFeature>) {
        self.state.lock().unwrap().features = features;
    }

    pub fn stream_requests(&self) -> StreamRequests {
        self.state.lock().unwrap().requested.clone()
    }

    /// Each stream gets the frame converted to the rate and channel count it asked for. Closed
    /// senders are pruned, so the returned count exposes a leaked subscription.
    pub fn push_frame(&self, frame: AudioFrame) -> usize {
        let mut state = self.state.lock().unwrap();
        state.streams.retain(|stream| {
            stream.tx.push(convert_frame(&frame, stream.sample_rate, stream.num_channels))
                != Pushed::Closed
        });
        state.streams.len()
    }

    /// Returns how many frames a reader got; zero means they were lost.
    pub fn push_frames(&self, frames: impl IntoIterator<Item = AudioFrame>) -> usize {
        frames.into_iter().map(|frame| usize::from(self.push_frame(frame) > 0)).sum()
    }

    /// The [`VideoFrameRef`] is built by the caller so a binding test can push a real host frame.
    /// Closed readers are pruned, so the returned count exposes a leaked subscription.
    pub fn push_video_frame(&self, frame: VideoFrameRef) -> usize {
        let mut state = self.state.lock().unwrap();
        state.video_streams.retain(|tx| tx.push(frame.clone()) != Pushed::Closed);
        state.video_streams.len()
    }

    pub(super) fn mark_subscribed(self: &Arc<Self>, subscribed: bool) {
        let was = self.subscribed.swap(subscribed, Ordering::SeqCst);
        if subscribed {
            // `notify_waiters` only wakes current waiters; later ones read the flag. Both are needed.
            self.subscription.notify_waiters();
        }
        let Some(participant) = self.owner.as_ref().and_then(Weak::upgrade) else { return };
        if subscribed && !was {
            self.bus.emit(RoomEvent::TrackSubscribed { participant, publication: self.clone() });
        } else if !subscribed && was {
            // As the SDK: the track's streams end, then `track_unsubscribed`.
            self.close_streams();
            self.bus.emit(RoomEvent::TrackUnsubscribed { participant, publication: self.clone() });
        }
    }

    pub(super) fn close_streams(&self) {
        self.subscribed.store(false, Ordering::SeqCst);
        // Dropping the senders (audio and video) ends `recv()`; these are the only copies.
        let mut state = self.state.lock().unwrap();
        state.streams.clear();
        state.video_streams.clear();
    }
}

impl TrackPublication for FakeTrackPublication {
    fn name(&self) -> &str {
        &self.name
    }

    fn sid(&self) -> &TrackSid {
        &self.sid
    }

    fn source(&self) -> TrackSource {
        self.source
    }

    fn kind(&self) -> i32 {
        match self.source {
            TrackSource::Camera | TrackSource::Screenshare => {
                lk_agents_abi::voice::rtc::TRACK_KIND_VIDEO
            }
            TrackSource::Microphone | TrackSource::ScreenshareAudio => {
                lk_agents_abi::voice::rtc::TRACK_KIND_AUDIO
            }
            TrackSource::Unknown => 0,
        }
    }

    fn is_subscribed(&self) -> Result<bool> {
        Ok(self.subscribed.load(Ordering::SeqCst))
    }

    fn audio_features(&self) -> Vec<AudioTrackFeature> {
        self.state.lock().unwrap().features.clone()
    }

    fn set_subscribed<'a>(&'a self, subscribed: bool) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            if self.owner.is_none() {
                return Err(RtcError::Unsupported("set_subscribed"));
            }
            // Recover the `Arc<Self>` through the owner rather than holding a self-weak.
            let me = self
                .owner
                .as_ref()
                .and_then(Weak::upgrade)
                .and_then(|p| {
                    p.state.lock().unwrap().publications.iter().find(|x| x.sid == self.sid).cloned()
                })
                .ok_or_else(|| RtcError::TrackGone(self.sid.clone()))?;
            me.mark_subscribed(subscribed);
            Ok(())
        })
    }

    fn wait_for_subscription<'a>(&'a self) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            if self.owner.is_some() {
                return Err(RtcError::Unsupported("wait_for_subscription"));
            }
            loop {
                let parked = self.subscription.notified();
                tokio::pin!(parked);
                // Register before checking, or a `mark_subscribed` in between is missed and this hangs.
                parked.as_mut().enable();
                if self.subscribed.load(Ordering::SeqCst) {
                    return Ok(());
                }
                parked.await;
            }
        })
    }

    fn open_audio_stream(&self, opts: AudioStreamOptions) -> Result<Arc<dyn AudioInputStream>> {
        Ok(Arc::new(FakeAudioInputStream { frames: Mutex::new(Some(self.audio_stream(opts))) }))
    }

    fn open_video_stream(&self) -> Result<Box<dyn StreamSource<VideoFrameRef>>> {
        Ok(Box::new(super::LossySource(self.video_stream())))
    }
}

impl FakeTrackPublication {
    /// `frame_size_ms` is ignored: one supplied frame remains one delivery in the fake.
    pub fn audio_stream(&self, opts: AudioStreamOptions) -> BoundedLossy<AudioFrame> {
        let (tx, rx) = BoundedLossy::audio();
        let mut state = self.state.lock().unwrap();
        state.streams.push(OpenStream {
            tx,
            sample_rate: Some(opts.sample_rate),
            num_channels: Some(opts.num_channels),
        });
        state.requested.0.push(opts);
        rx
    }

    pub fn video_stream(&self) -> BoundedLossy<VideoFrameRef> {
        let (tx, rx) = BoundedLossy::channel(lk_agents_abi::voice::rtc::VIDEO_QUEUE_FRAMES);
        self.state.lock().unwrap().video_streams.push(tx);
        rx
    }
}

struct FakeAudioInputStream {
    frames: Mutex<Option<BoundedLossy<AudioFrame>>>,
}

impl AudioInputStream for FakeAudioInputStream {
    fn frames(&self) -> Result<Box<dyn StreamSource<AudioFrame>>> {
        let frames =
            self.frames.lock().unwrap().take().ok_or(RtcError::Closed("audio input stream"))?;
        Ok(Box::new(super::LossySource(frames)))
    }
}

/// What `rtc.AudioStream` does natively: remix, then resample (linear; only the declared rate and
/// length matter here). Degenerate inputs from a host object pass through untouched.
fn convert_frame(
    frame: &AudioFrame,
    want_rate: Option<u32>,
    want_channels: Option<u16>,
) -> AudioFrame {
    let (src_rate, src_ch) = (frame.sample_rate, frame.num_channels);
    let rate = want_rate.unwrap_or(src_rate);
    let channels = want_channels.unwrap_or(src_ch);
    if (rate == src_rate && channels == src_ch)
        || src_rate == 0
        || rate == 0
        || src_ch == 0
        || channels == 0
    {
        return frame.clone();
    }

    let src_ch = usize::from(src_ch);
    let channels_n = usize::from(channels);
    let src: Vec<i16> =
        frame.data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
    let src_n = std::cmp::min(frame.samples_per_channel as usize, src.len() / src_ch);
    if src_n == 0 {
        return frame.clone();
    }

    // `round`: a rate that does not divide evenly is off by at most half a sample.
    let out_n = std::cmp::max(
        1,
        ((src_n as u64 * u64::from(rate)) as f64 / f64::from(src_rate)).round() as usize,
    );

    // To mono averages channels, from mono replicates, otherwise the last channel repeats.
    let mixed = |s: usize, c: usize| -> i16 {
        if channels_n == 1 && src_ch > 1 {
            let sum: i32 = (0..src_ch).map(|k| i32::from(src[s * src_ch + k])).sum();
            (sum / src_ch as i32) as i16
        } else {
            src[s * src_ch + std::cmp::min(c, src_ch - 1)]
        }
    };

    let mut out = Vec::with_capacity(out_n * channels_n * 2);
    for i in 0..out_n {
        // The ratio, not `(src_n-1)/(out_n-1)`, so consecutive frames line up.
        let pos = i as f64 * f64::from(src_rate) / f64::from(rate);
        let lo = pos.floor() as usize;
        let hi = std::cmp::min(lo + 1, src_n - 1);
        let t = pos - lo as f64;
        let lo = std::cmp::min(lo, src_n - 1);
        for c in 0..channels_n {
            let a = f64::from(mixed(lo, c));
            let b = f64::from(mixed(hi, c));
            let v = (a + (b - a) * t).round().clamp(f64::from(i16::MIN), f64::from(i16::MAX));
            out.extend_from_slice(&(v as i16).to_le_bytes());
        }
    }

    AudioFrame::new(out, rate, channels, out_n as u32)
}

/// Subscribe a local publication as a remote peer would; releases `wait_for_subscription`.
pub fn subscribe_local(publication: &Arc<FakeTrackPublication>) {
    publication.mark_subscribed(true);
}
