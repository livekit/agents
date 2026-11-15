//! A socket TTS stream's segments: the machine decides when a segment opens, ends, plays and
//! completes, and when the read deadline runs; `TtsSegments` applies its effects to the emitter
//! and the request's deadline.

use std::collections::VecDeque;
use std::sync::Arc;

use lk_agents_abi::TranscriptChunk;
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::tts::segments::{
    self as abi, SegmentEnd, SegmentError, SegmentInput, SegmentOutput, SegmentStart, SegmentText,
};
use lk_agents_abi::tts::stream_pacer::PushedAudio;
use lk_agents_abi::tts::types::SpeechChunk;
use lk_agents_abi::types::provider::ReadDeadline;
use lk_agents_abi::usage::TokenUsage;
use lk_agents_net::signal;
use parking_lot::Mutex;

/// What the segments do to the emitter and the read deadline, in order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SegmentEffect {
    StartSegment(String),
    EndSegment,
    Push(Vec<u8>),
    Words(Vec<TranscriptChunk>),
    Usage(Option<TokenUsage>, Option<u64>),
    ExpectResponse,
    Progress,
    CompleteResponse,
    /// The provider ended this segment while it waited for more text.
    Expired(String),
}

/// Audio, words or usage a segment received before it played.
#[derive(Debug)]
enum Held {
    Audio(Vec<u8>),
    Words(Vec<TranscriptChunk>),
    Usage(Option<TokenUsage>, Option<u64>),
}

#[derive(Debug)]
struct Segment {
    id: String,
    /// Its text ended: its answer is owed.
    ended: bool,
    done: bool,
    held: Vec<Held>,
}

/// A segment report that breaks the protocol: it names no segment open on this stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Unknown;

/// One attempt's segments, in order. The first is the one playing.
#[derive(Debug, Default)]
pub(crate) struct SegmentMachine {
    queued: VecDeque<SegmentInput>,
    /// The segment taking text.
    open: Option<String>,
    playing: VecDeque<Segment>,
}

impl SegmentMachine {
    /// A chunk of input; `id` names the segment it opens when none is open.
    pub(crate) fn push(
        &mut self,
        chunk: SpeechChunk,
        id: impl FnOnce() -> String,
        effects: &mut Vec<SegmentEffect>,
    ) {
        let segment_id = match &self.open {
            Some(open) => open.clone(),
            None => {
                let segment_id = id();
                if self.playing.is_empty() {
                    effects.push(SegmentEffect::StartSegment(segment_id.clone()));
                }
                self.playing.push_back(Segment {
                    id: segment_id.clone(),
                    ended: false,
                    done: false,
                    held: Vec::new(),
                });
                self.queued.push_back(SegmentInput::Start(SegmentStart {
                    segment_id: segment_id.clone(),
                }));
                self.open = Some(segment_id.clone());
                segment_id
            }
        };
        self.queued.push_back(SegmentInput::Text(SegmentText { segment_id, chunk }));
    }

    /// A flush: the open segment's text ends. A flush with no text opens nothing.
    pub(crate) fn flush(&mut self) {
        if let Some(segment_id) = self.open.take() {
            self.queued.push_back(SegmentInput::End(SegmentEnd { segment_id }));
        }
    }

    /// The next input `encode` takes. A segment's end arms the read deadline.
    pub(crate) fn next(&mut self, effects: &mut Vec<SegmentEffect>) -> Option<SegmentInput> {
        let item = self.queued.pop_front()?;
        if let SegmentInput::End(end) = &item
            && let Some(segment) = self.segment(&end.segment_id)
        {
            segment.ended = true;
            effects.push(SegmentEffect::ExpectResponse);
        }
        Some(item)
    }

    /// What the provider answered for a segment; `id` names the segment that takes the rest of
    /// a segment's text when the provider ended it early.
    pub(crate) fn deliver(
        &mut self,
        output: SegmentOutput,
        id: impl FnOnce() -> String,
        effects: &mut Vec<SegmentEffect>,
    ) -> Result<(), Unknown> {
        let (segment_id, held) = match output {
            SegmentOutput::Audio(audio) => (audio.segment_id, Held::Audio(audio.data)),
            SegmentOutput::Words(words) => (words.segment_id, Held::Words(words.words)),
            SegmentOutput::Usage(usage) => {
                (usage.segment_id, Held::Usage(usage.usage, usage.characters_count))
            }
            SegmentOutput::Done(done) => return self.done(&done.segment_id, id, effects),
        };
        let head = self.playing.front().is_some_and(|s| s.id == segment_id);
        let segment = self.segment(&segment_id).ok_or(Unknown)?;
        effects.push(SegmentEffect::Progress);
        match head {
            true => effects.push(played(held)),
            false => segment.held.push(held),
        }
        Ok(())
    }

    /// The provider finished the session the ended segments went on: each one not yet done is.
    pub(crate) fn complete_ended(&mut self, effects: &mut Vec<SegmentEffect>) {
        let ended: Vec<String> =
            self.playing.iter().filter(|s| s.ended && !s.done).map(|s| s.id.clone()).collect();
        for segment_id in ended {
            let _ = self.done(&segment_id, String::new, effects);
        }
    }

    pub(crate) fn holds(&self, segment_id: &str) -> bool {
        self.playing.iter().any(|s| s.id == segment_id && !s.done)
    }

    pub(crate) fn open(&self) -> Vec<String> {
        self.playing.iter().filter(|s| !s.done).map(|s| s.id.clone()).collect()
    }

    /// Every segment is done.
    pub(crate) fn settled(&self) -> bool {
        self.playing.is_empty()
    }

    fn segment(&mut self, segment_id: &str) -> Option<&mut Segment> {
        self.playing.iter_mut().find(|s| s.id == segment_id && !s.done)
    }

    /// The provider finished `segment_id`. One whose text had not ended was ended by the provider
    /// while it waited for more (its context expired while the LLM was slow): it is done with what
    /// it said, and the rest of its text opens a new segment, a new context.
    fn done(
        &mut self,
        segment_id: &str,
        id: impl FnOnce() -> String,
        effects: &mut Vec<SegmentEffect>,
    ) -> Result<(), Unknown> {
        let segment = self.segment(segment_id).ok_or(Unknown)?;
        let expired = !segment.ended;
        segment.done = true;
        if expired {
            effects.push(SegmentEffect::Expired(segment_id.to_owned()));
            self.reopen(segment_id, id);
        }
        while self.playing.front().is_some_and(|s| s.done) {
            self.playing.pop_front();
            effects.push(SegmentEffect::EndSegment);
            if let Some(next) = self.playing.front_mut() {
                effects.push(SegmentEffect::StartSegment(next.id.clone()));
                effects.extend(next.held.drain(..).map(played));
            }
        }
        if !self.playing.iter().any(|s| s.ended && !s.done) {
            effects.push(SegmentEffect::CompleteResponse);
        }
        Ok(())
    }

    /// Moves the text `expired` has yet to send to a new segment right after it: its queued
    /// text, its end, and the text still to come if it is the open one. An end with no text
    /// left to carry is dropped.
    fn reopen(&mut self, expired: &str, id: impl FnOnce() -> String) {
        let carried = |item: &SegmentInput| match item {
            SegmentInput::Text(text) => text.segment_id == expired,
            SegmentInput::Start(_) | SegmentInput::End(_) => false,
        };
        let Some(first) = self.queued.iter().position(carried) else {
            self.queued.retain(
                |item| !matches!(item, SegmentInput::End(end) if end.segment_id == expired),
            );
            if self.open.as_deref() == Some(expired) {
                self.open = None;
            }
            return;
        };
        let segment_id = id();
        for item in self.queued.iter_mut() {
            match item {
                SegmentInput::Text(SegmentText { segment_id: owner, .. })
                | SegmentInput::End(SegmentEnd { segment_id: owner })
                    if owner == expired =>
                {
                    owner.clone_from(&segment_id);
                }
                SegmentInput::Start(_) | SegmentInput::Text(_) | SegmentInput::End(_) => {}
            }
        }
        self.queued
            .insert(first, SegmentInput::Start(SegmentStart { segment_id: segment_id.clone() }));
        if self.open.as_deref() == Some(expired) {
            self.open = Some(segment_id.clone());
        }
        let at = self.playing.iter().position(|s| s.id == expired).map_or(0, |i| i + 1);
        let segment = Segment { id: segment_id, ended: false, done: false, held: Vec::new() };
        self.playing.insert(at, segment);
    }
}

fn played(held: Held) -> SegmentEffect {
    match held {
        Held::Audio(data) => SegmentEffect::Push(data),
        Held::Words(words) => SegmentEffect::Words(words),
        Held::Usage(usage, characters) => SegmentEffect::Usage(usage, characters),
    }
}

/// `_core.TtsSegments`.
pub struct TtsSegments {
    emitter: Arc<dyn PushedAudio>,
    deadline: Arc<dyn ReadDeadline>,
    label: String,
    machine: Mutex<SegmentMachine>,
    settled: signal::Shared<bool>,
}

impl TtsSegments {
    /// Runs `op` on the machine and applies its effects with no lock held.
    fn run<T>(
        &self,
        op: impl FnOnce(&mut SegmentMachine, &mut Vec<SegmentEffect>) -> T,
    ) -> Result<T, SegmentError> {
        let mut effects = Vec::new();
        let (out, settled) = {
            let mut machine = self.machine.lock();
            let out = op(&mut machine, &mut effects);
            (out, machine.settled())
        };
        self.settled.modify(|s| *s = settled);
        for effect in effects {
            match effect {
                SegmentEffect::StartSegment(id) => self.emitter.start_segment(id)?,
                SegmentEffect::EndSegment => self.emitter.end_segment()?,
                SegmentEffect::Push(data) => self.emitter.push(&data)?,
                SegmentEffect::Words(words) => self.emitter.push_timed_transcript(words)?,
                SegmentEffect::Usage(usage, characters) => {
                    self.emitter.report_usage(usage, characters)?;
                }
                SegmentEffect::ExpectResponse => self.deadline.expect_response(),
                SegmentEffect::Progress => self.deadline.progress(),
                SegmentEffect::CompleteResponse => self.deadline.complete_response(),
                SegmentEffect::Expired(id) => log::warn!(target: lk_agents_abi::LOG_TARGET,
                    "{}: the provider ended segment {id} while the LLM was slow to send its next \
                     text (its context expired); the rest of its text goes on a new context",
                    self.label),
            }
        }
        Ok(out)
    }
}

impl abi::TtsSegments for TtsSegments {
    fn new(emitter: Arc<dyn PushedAudio>, deadline: Arc<dyn ReadDeadline>, label: String) -> Self {
        TtsSegments {
            emitter,
            deadline,
            label,
            machine: Mutex::default(),
            settled: signal::Shared::new(true),
        }
    }

    fn push(&self, chunk: SpeechChunk) -> Result<(), SegmentError> {
        self.run(|m, effects| m.push(chunk, || crate::shortuuid(""), effects))
    }

    fn flush(&self) {
        self.machine.lock().flush();
    }

    fn next(&self) -> Result<Option<SegmentInput>, SegmentError> {
        self.run(|m, effects| m.next(effects))
    }

    fn deliver(&self, output: SegmentOutput) -> Result<(), SegmentError> {
        let shown = format!("{output:?}");
        match self.run(|m, effects| m.deliver(output, || crate::shortuuid(""), effects))? {
            Ok(()) => Ok(()),
            Err(Unknown) => {
                let message = format!("{} sent a message that breaks its protocol", self.label);
                let error = ApiError::protocol(message, format!("{shown} names no open segment"));
                Err(SegmentError::Protocol(error.into()))
            }
        }
    }

    fn complete_ended(&self) -> Result<(), SegmentError> {
        self.run(SegmentMachine::complete_ended)
    }

    fn holds(&self, segment_id: String) -> bool {
        self.machine.lock().holds(&segment_id)
    }

    fn open(&self) -> Vec<String> {
        self.machine.lock().open()
    }

    async fn settled(&self) {
        self.settled.wait_until(|settled| settled.then_some(())).await;
    }
}

#[cfg(test)]
#[path = "segments_tests.rs"]
mod tests;
