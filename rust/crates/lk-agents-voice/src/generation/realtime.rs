//! Realtime message ordering, output modality and the server's retained transcript.

use crate::agent_activity::realtime::response_id_of;
use crate::transcription::Heard;
use lk_agents_abi::AudioFrame;
use lk_agents_abi::llm::realtime::{self as rt, Event, Modalities};
use std::time::Duration;

/// How many of a speech's events its player holds ahead of the playout: the server sends a
/// reply faster than it plays, so this is a whole reply (minutes of the smallest audio deltas).
/// A full player is overrun: the runtime stops delivering to it and its speech ends on what it
/// holds.
pub const PLAYER_CAPACITY: usize = 8192;
pub struct RealtimePlayout {
    current: Option<usize>,
    segments: Vec<String>,
    response_id: Option<String>,
    message_id: Option<String>,
    messages: Vec<Message>,
    audio: Duration,
    received: bool,
    finished: bool,
    heard: Heard,
    segment_start: usize,
    has_audio_output: bool,
}

pub struct RealtimeObservation {
    pub interrupted: bool,
    pub synthesizing: bool,
    pub tts_available: bool,
}

pub enum RealtimeEffect {
    OpenText,
    Text(String),
    Audio(AudioFrame),
    VoiceText { segment: usize, text: String },
    Boundary,
    End,
    ToolCall(rt::FunctionCallReady),
    Failed(String),
}

pub enum RealtimeControl {
    ToolCall(rt::FunctionCallReady),
    End,
    Defer(Event),
}

pub struct RealtimeTranscript {
    pub heard: Heard,
    pub segments: Vec<String>,
    pub response_id: Option<String>,
    pub message_id: Option<String>,
    pub received: bool,
}

struct Message {
    id: String,
    start: Duration,
    audio: Duration,
    modalities: Option<Modalities>,
    done: bool,
}

impl Default for RealtimePlayout {
    fn default() -> Self {
        Self {
            current: None,
            segments: Vec::new(),
            response_id: None,
            message_id: None,
            messages: Vec::new(),
            audio: Duration::ZERO,
            received: false,
            finished: false,
            heard: Heard::complete(String::new(), Duration::ZERO, 0),
            segment_start: 0,
            has_audio_output: false,
        }
    }
}

impl RealtimePlayout {
    /// A refused binding ends the reply on what its preceding bindings played.
    pub fn binding_refused(&mut self) {
        self.heard.complete = false;
    }

    /// Each binding contributes its receipt once, including a failed forwarding's partial receipt.
    pub fn played(&mut self, part: Heard) {
        self.heard.text.push_str(&part.text);
        self.heard.position += part.position;
        self.heard.played += part.played;
        self.heard.frames += part.frames;
        self.heard.complete &= part.complete;
    }

    pub fn finished(&self) -> bool {
        self.finished
    }

    /// No later message is admitted after a cut, provider completion or speech interruption.
    pub fn stop_playout(&self, interrupted: bool) -> bool {
        !self.heard.complete || self.finished || interrupted
    }

    pub fn interrupted(&self, interrupted: bool) -> bool {
        !self.heard.complete || interrupted
    }

    /// An admitted binding freezes its output facts until the message settles.
    pub fn bind(&mut self, has_audio_output: bool) {
        self.has_audio_output = has_audio_output;
        self.segment_start = self.segments.len();
    }

    pub fn has_audio_output(&self) -> bool {
        self.has_audio_output
    }

    /// A message that played no audio but forwarded words ends the bound audio leg, as the
    /// pipeline's transcript-only segment does, once the text chain `delivered` them. One with no
    /// words ends nothing: its text leg never opened a segment, and an audio end alone would open
    /// one the next speech's words pair with; nor does one whose chain was cancelled with the
    /// words still in it (a cut `transcription_node`).
    pub fn end_unvoiced(&self, heard: &Heard, has_text_output: bool, delivered: bool) -> bool {
        heard.frames == 0
            && heard.complete
            && has_text_output
            && delivered
            && !self.text().is_empty()
            && self.has_audio_output
    }

    /// Whether the current binding played a message. One the generation ended before any
    /// message still closes its transcript chain, or the chain's drain never ends.
    pub fn carried_message(&self) -> bool {
        self.segments.len() > self.segment_start
    }

    pub fn closed(&mut self) {
        self.finished = true;
    }

    pub fn text(&self) -> String {
        self.segments[self.segment_start..].concat()
    }

    pub fn into_transcript(self) -> RealtimeTranscript {
        RealtimeTranscript {
            heard: self.heard,
            segments: self.segments,
            response_id: self.response_id,
            message_id: self.message_id,
            received: self.received,
        }
    }

    fn note_event(&mut self, event: &Event) {
        self.received = true;
        if self.response_id.is_none()
            && let Some(id) = response_id_of(event)
        {
            self.response_id = Some(id.to_owned());
        }
    }

    /// The next provider event, admitted for the current output binding.
    pub fn handle(
        &mut self,
        event: Event,
        observed: RealtimeObservation,
    ) -> Option<RealtimeEffect> {
        self.note_event(&event);
        match event {
            Event::MessageStarted(message) => {
                self.message_started(message.message_id.clone());
                self.message_id.get_or_insert(message.message_id);
                self.current = Some(self.segments.len());
                self.segments.push(String::new());
                Some(RealtimeEffect::OpenText)
            }
            Event::TextDelta(delta) => {
                let text = self.current.and_then(|i| self.segments.get_mut(i))?;
                text.push_str(&delta.delta);
                (!observed.interrupted).then_some(RealtimeEffect::Text(delta.delta))
            }
            Event::AudioDelta(delta) => {
                self.add_audio(&delta.frame);
                Some(RealtimeEffect::Audio(delta.frame))
            }
            Event::MessageModalities(message) => {
                self.settle(&message.message_id, message.modalities);
                if message.modalities != Modalities::TextOnly
                    || !self.has_audio_output
                    || !observed.tts_available
                    || observed.interrupted
                    || observed.synthesizing
                {
                    return None;
                }
                let segment = self.current?;
                Some(RealtimeEffect::VoiceText {
                    segment,
                    text: self.segments.get(segment).cloned().unwrap_or_default(),
                })
            }
            Event::MessageDone(message) => {
                self.message_done(&message.message_id);
                self.current = None;
                Some(RealtimeEffect::Boundary)
            }
            Event::FunctionCall(call) => Some(RealtimeEffect::ToolCall(call)),
            Event::GenerationDone(_) => {
                self.finished = true;
                Some(RealtimeEffect::End)
            }
            Event::GenerateReplyFailed(failed) => {
                Some(RealtimeEffect::Failed(failed.error.message))
            }
            _ => None,
        }
    }

    /// While a message awaits its playback receipt, media for the next binding stays queued.
    pub fn control(&mut self, event: Event) -> Option<RealtimeControl> {
        self.note_event(&event);
        match event {
            Event::FunctionCall(call) => Some(RealtimeControl::ToolCall(call)),
            Event::GenerationDone(_) => {
                self.finished = true;
                Some(RealtimeControl::End)
            }
            Event::MessageModalities(message) => {
                self.settle(&message.message_id, message.modalities);
                None
            }
            event @ (Event::MessageStarted(_)
            | Event::TextDelta(_)
            | Event::AudioDelta(_)
            | Event::MessageDone(_)
            | Event::GenerateReplyFailed(_)) => Some(RealtimeControl::Defer(event)),
            _ => None,
        }
    }

    /// Text arriving while the synthesized audio is awaited still belongs to the open message.
    pub fn synthesized_text(&mut self, delta: &str) {
        if let Some(text) = self.current.and_then(|i| self.segments.get_mut(i)) {
            text.push_str(delta);
        }
    }

    pub fn add_audio(&mut self, frame: &AudioFrame) {
        let len = lk_agents_abi::voice::rtc::frame_duration(frame);
        self.audio += len;
        if let Some(message) = self.messages.last_mut() {
            message.audio += len;
        }
    }

    /// Already queued server media contributes to truncation even when it was never played.
    pub fn unplayed(&mut self, event: Event) {
        match event {
            Event::MessageStarted(message) => self.message_started(message.message_id),
            Event::MessageModalities(message) => {
                self.settle(&message.message_id, message.modalities)
            }
            Event::AudioDelta(delta) => self.add_audio(&delta.frame),
            Event::MessageDone(message) => self.message_done(&message.message_id),
            _ => {}
        }
    }

    fn message_started(&mut self, id: String) {
        self.messages.push(Message {
            id,
            start: self.audio,
            audio: Duration::ZERO,
            modalities: None,
            done: false,
        });
    }

    fn message_done(&mut self, id: &str) {
        if let Some(message) = self.messages.iter_mut().find(|message| message.id == id) {
            message.done = true;
        }
    }

    fn settle(&mut self, id: &str, modalities: Modalities) {
        if let Some(message) = self.messages.iter_mut().find(|message| message.id == id) {
            message.modalities = Some(modalities);
        }
    }

    /// A failure settles the server's transcript at the last observed playback position.
    pub fn heard_until(&mut self) -> Vec<rt::Truncate> {
        self.heard.complete = false;
        self.cuts()
    }

    /// Keep complete audio messages; cut partial audio and text to what the person heard.
    pub fn cuts(&self) -> Vec<rt::Truncate> {
        let heard = &self.heard;
        let mut words = heard.text.as_str();
        let mut cuts = Vec::new();
        for (index, message) in self.messages.iter().enumerate() {
            let text = self.segments.get(index).map_or("", String::as_str);
            let (said, rest) = split_chars(words, text.chars().count());
            words = rest;
            let has_audio = message
                .modalities
                .map_or(message.audio > Duration::ZERO || self.has_audio_output, |modalities| {
                    modalities == Modalities::AudioAndText
                });
            if has_audio {
                let ms =
                    |duration: Duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX);
                let heard_ms = ms(heard.played.saturating_sub(message.start).min(message.audio));
                if message.done
                    && message.audio > Duration::ZERO
                    && heard_ms + 1 >= ms(message.audio)
                {
                    continue;
                }
                cuts.push(rt::Truncate {
                    message_id: message.id.clone(),
                    has_audio: true,
                    audio_end_ms: heard_ms,
                    audio_transcript: None,
                });
            } else if said.len() < text.len() {
                cuts.push(rt::Truncate {
                    message_id: message.id.clone(),
                    has_audio: false,
                    audio_end_ms: 0,
                    audio_transcript: Some(said.to_owned()),
                });
            }
        }
        cuts
    }
}

fn split_chars(text: &str, count: usize) -> (&str, &str) {
    let at = text.char_indices().nth(count).map_or(text.len(), |(index, _)| index);
    text.split_at(at)
}

/// A reply interrupted before authorization has no heard message.
pub fn unheard_messages(events: impl IntoIterator<Item = Event>) -> Vec<rt::Truncate> {
    events
        .into_iter()
        .filter_map(|event| match event {
            Event::MessageStarted(message) => Some(rt::Truncate {
                message_id: message.message_id,
                has_audio: true,
                audio_end_ms: 0,
                audio_transcript: None,
            }),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observed() -> RealtimeObservation {
        RealtimeObservation { interrupted: false, synthesizing: false, tts_available: false }
    }

    // Catches: a message the server cancelled after starting it, before any of its words, ended
    // the bound audio leg as an unvoiced segment. Its text leg had opened no segment, so the
    // audio end alone opened one, and the next speech's words paired with it and were shown whole
    // before their audio (`transcript_follows_audio`, survey seed 3465).
    #[test]
    fn only_a_message_that_forwarded_words_ends_the_audio_leg_unvoiced() {
        let started = |id: &str| {
            Event::MessageStarted(rt::MessageStarted { message_id: id.into(), response_id: None })
        };
        let mut playout = RealtimePlayout::default();
        playout.bind(true);
        let _ = playout.handle(started("m1"), observed());
        let silent = Heard::complete(String::new(), Duration::ZERO, 0);
        assert!(!playout.end_unvoiced(&silent, true, true), "a message with no words ends nothing");

        let delta = rt::TextDelta {
            message_id: "m1".into(),
            response_id: None,
            delta: "Hello.".into(),
            start_time: None,
        };
        let _ = playout.handle(Event::TextDelta(delta), observed());
        assert!(playout.end_unvoiced(&silent, true, true), "its words end the audio leg with none");
    }
}
