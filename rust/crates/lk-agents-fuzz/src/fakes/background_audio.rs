//! Background track execution over fake room and stream ports. Selection, fades and mixing use
//! the real decision code; ambient and thinking sources keep independent playback lifetimes.

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::background_audio::{FrameStream, PlaybackError, PlaybackInput, Sound};
use lk_agents_abi::voice::room_owner::SessionRoom;
use lk_agents_abi::voice::rtc::{AudioSource, RtcError, TrackPublishOptions};
use lk_agents_abi::{AudioFrame, BoxFuture, Spawner, SpawnerExt};
use lk_agents_media::{Mixer, StreamId};
use lk_agents_voice::background_audio::{
    AUDIO_SOURCE_BUFFER_MS, Playback, TRACK_NAME, apply_gain, select_sound,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, watch};

pub struct FakeBackgroundAudioPlayer {
    spawner: Arc<dyn Spawner>,
    output: Mutex<Option<Output>>,
    control: Mutex<Control>,
}

#[derive(Default)]
struct Control {
    play: Option<mpsc::UnboundedSender<Source>>,
    thinking: Option<Arc<Handle>>,
}

#[derive(Clone)]
struct Output {
    stop: watch::Sender<bool>,
    done: watch::Receiver<bool>,
}

struct Handle {
    stopped: AtomicBool,
    done: AtomicBool,
    fade_out: f64,
}

impl Handle {
    fn stop(&self) {
        if !self.done.load(Ordering::Acquire) {
            self.stopped.store(true, Ordering::Release);
            if self.fade_out <= 0.0 {
                self.done.store(true, Ordering::Release);
            }
        }
    }
}

struct Source {
    input: Input,
    playback: Playback,
    handle: Arc<Handle>,
}

enum Input {
    Reading {
        stream: Arc<dyn FrameStream>,
        pending: Option<BoxFuture<'static, Result<Option<AudioFrame>, ProviderError>>>,
    },
    Closing(BoxFuture<'static, ()>),
    Closed,
}

impl Source {
    fn selected(sounds: Vec<Sound>, draw: f64, choose: bool) -> Option<Self> {
        let configs: Vec<_> = sounds.iter().map(|sound| sound.config.clone()).collect();
        let index =
            if choose { select_sound(&configs, draw) } else { (!sounds.is_empty()).then_some(0) }?;
        let sound = sounds.into_iter().nth(index)?;
        // The survey supplies controlled finite streams; file decoding has runtime conformance tests.
        let PlaybackInput::Stream(stream) = sound.input else {
            panic!("background survey requires a controlled host stream");
        };
        let handle = Arc::new(Handle {
            stopped: false.into(),
            done: false.into(),
            fade_out: sound.config.fade_out,
        });
        Some(Self {
            input: Input::Reading { stream, pending: None },
            playback: Playback::new(&sound.config),
            handle,
        })
    }

    async fn next(&mut self) -> Result<Option<AudioFrame>, tokio::time::error::Elapsed> {
        let Input::Reading { stream, pending } = &mut self.input else { return Ok(None) };
        if pending.is_none() {
            let stream = stream.clone();
            *pending = Some(Box::pin(async move { stream.next().await }));
        }
        let result =
            tokio::time::timeout(Duration::from_millis(200), pending.as_mut().unwrap()).await?;
        *pending = None;
        Ok(result.ok().flatten())
    }

    async fn close(&mut self) {
        self.input = match std::mem::replace(&mut self.input, Input::Closed) {
            Input::Reading { stream, pending } => {
                stream.cancel_pending();
                Input::Closing(Box::pin(async move {
                    if let Some(pending) = pending {
                        let _ = pending.await;
                    }
                    stream.close().await;
                }))
            }
            input => input,
        };
        // The source retains this future if the mixer's await is cancelled by a stop.
        if let Input::Closing(close) = &mut self.input {
            close.await;
        }
        self.input = Input::Closed;
        self.handle.done.store(true, Ordering::Release);
    }
}

impl FakeBackgroundAudioPlayer {
    pub fn new(spawner: Arc<dyn Spawner>) -> Self {
        Self { spawner, output: Mutex::new(None), control: Mutex::new(Control::default()) }
    }

    pub async fn start(
        &self,
        room: Arc<dyn SessionRoom>,
        ambient: Vec<Sound>,
        draw: f64,
    ) -> Result<(), RtcError> {
        let participant = room.rtc().ok_or(RtcError::NotConnected)?.local().clone();
        let (source, publication) = participant
            .publish_audio_track(TrackPublishOptions {
                name: TRACK_NAME.into(),
                sample_rate: super::SAMPLE_RATE,
                num_channels: 1,
                queue_size_ms: AUDIO_SOURCE_BUFFER_MS,
                ..Default::default()
            })
            .await?;
        let (play, rx) = mpsc::unbounded_channel();
        if let Some(source) = Source::selected(ambient, draw, true) {
            let _ = play.send(source);
        }
        let (stop, stopped) = watch::channel(false);
        let (done, finished) = watch::channel(false);
        let sid = publication.sid().clone();
        self.spawner.spawn_unboxed(async move {
            pump(source.clone(), rx, stopped).await;
            let _ = source.close().await;
            let _ = participant.unpublish_track(&sid).await;
            done.send_replace(true);
        });
        self.control.lock().unwrap().play = Some(play);
        *self.output.lock().unwrap() = Some(Output { stop, done: finished });
        Ok(())
    }

    pub fn on_agent_state_for_host(
        &self,
        sounds: Vec<Sound>,
        draw: f64,
        thinking: bool,
        choose: bool,
    ) -> Result<(), PlaybackError> {
        if sounds.is_empty() {
            return Ok(());
        }
        let mut control = self.control.lock().unwrap();
        let Some(play) = control.play.clone() else { return Ok(()) };
        if thinking {
            if control.thinking.as_ref().is_some_and(|handle| !handle.done.load(Ordering::Acquire))
            {
                return Ok(());
            }
            if let Some(source) = Source::selected(sounds, draw, choose) {
                control.thinking = Some(source.handle.clone());
                let _ = play.send(source);
            }
        } else if let Some(handle) = &control.thinking {
            handle.stop();
        }
        Ok(())
    }

    pub async fn close(&self) {
        self.control.lock().unwrap().play = None;
        let output = self.output.lock().unwrap().clone();
        if let Some(mut output) = output {
            output.stop.send_replace(true);
            let _ = output.done.wait_for(|done| *done).await;
        }
    }
}

async fn round(mixer: &mut Mixer, sources: &mut BTreeMap<StreamId, Source>) -> Option<Vec<i16>> {
    for id in mixer.stream_ids() {
        if sources.get(&id).is_some_and(|source| source.handle.done.load(Ordering::Acquire)) {
            mixer.remove_stream(id);
            if let Some(source) = sources.get_mut(&id) {
                source.close().await;
            }
            sources.remove(&id);
            continue;
        }
        while mixer.wants_more(id) {
            let Some(source) = sources.get_mut(&id) else { break };
            let frame = match source.next().await {
                Ok(Some(frame)) => frame,
                Ok(None) => {
                    mixer.mark_exhausted(id);
                    break;
                }
                Err(_) => {
                    mixer.note_timeout(id);
                    break;
                }
            };
            let pcm: Vec<_> = frame
                .data
                .chunks_exact(2)
                .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
                .collect();
            let step = source.playback.on_frame(
                frame.samples_per_channel as usize,
                frame.sample_rate,
                source.handle.stopped.load(Ordering::Acquire),
            );
            let pcm = match step.gain {
                Some(gain) => apply_gain(&pcm, &gain, frame.num_channels),
                None => pcm,
            };
            mixer.push(id, &pcm);
            if step.last {
                mixer.mark_exhausted(id);
                break;
            }
        }
    }
    let block = mixer.mix();
    for id in mixer.drain_removals() {
        if let Some(source) = sources.get_mut(&id) {
            source.close().await;
        }
        sources.remove(&id);
    }
    block
}

async fn pump(
    sink: Arc<dyn AudioSource>,
    mut play: mpsc::UnboundedReceiver<Source>,
    mut stop: watch::Receiver<bool>,
) {
    let mut mixer = Mixer::new(super::SAMPLE_RATE, 1, 2_400);
    let mut sources = BTreeMap::new();
    loop {
        while let Ok(source) = play.try_recv() {
            sources.insert(mixer.add_stream(), source);
        }
        if sources.is_empty() {
            tokio::select! {
                biased;
                _ = stop.wait_for(|closed| *closed) => break,
                source = play.recv() => match source {
                    Some(source) => { sources.insert(mixer.add_stream(), source); }
                    None => break,
                },
            }
            continue;
        }
        let pcm = tokio::select! {
            biased;
            _ = stop.wait_for(|closed| *closed) => break,
            pcm = round(&mut mixer, &mut sources) => pcm,
        };
        if let Some(pcm) = pcm {
            let data: Vec<u8> = pcm.into_iter().flat_map(i16::to_le_bytes).collect();
            let frame = AudioFrame::new(data, super::SAMPLE_RATE, 1, 2_400);
            tokio::select! {
                biased;
                _ = stop.wait_for(|closed| *closed) => break,
                result = sink.capture(&frame) => if result.is_err() { break },
            }
        } else if !sources.is_empty() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }
    play.close();
    while let Ok(mut source) = play.try_recv() {
        source.close().await;
    }
    for (_, mut source) in sources {
        source.close().await;
    }
}

#[cfg(test)]
mod tests;
