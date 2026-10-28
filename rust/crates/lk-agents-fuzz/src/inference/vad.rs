//! One VAD stream (`lk_agents_inference::vad::machine::VadMachine`) fed a microphone and answered
//! by a fake model runtime, with the host flushing it and updating its options midway.
//!
//! The microphone delivers 10 or 20 ms frames at 16, 24, 44.1 or 48 kHz, mono or stereo, in real
//! time; now and then a frame at another rate (a track switched under the stream). The model runs
//! one window at a time, as the runtime's media worker does: a window it was handed is scored
//! after 1-5 ms, or 50-400 ms when the worker is contended (past the slow-inference threshold),
//! on a speech-like probability stream: spoken runs and silences of drawn lengths, flickering
//! around the thresholds. A flush resets the stream while a window may still be scoring; its
//! result arrives afterwards anyway. The recording corpus has no VAD timings yet: the latencies
//! are drawn, not calibrated.
//!
//! The rules, over the trace:
//! * `vad_one_window_in_flight`: no window is handed out while another awaits its result;
//! * `vad_stale_result_is_inert`: a result for a window not in flight (cut by a flush) changes
//!   nothing: no event and no next window;
//! * `vad_sample_index_advances_once_per_result`: each accepted result yields one
//!   `inference_done`, its sample index one window past the last since the flush, and its edges
//!   carry that index;
//! * `vad_speech_edges_alternate`: `start_of_speech` and `end_of_speech` alternate from a start,
//!   and every `inference_done` reports the speaking state they left.

use std::collections::BTreeSet;

use fastrand::Rng;
use lk_agents_abi::{VadEvent, VadEventType};
use lk_agents_inference::abi::AudioFrame;
use lk_agents_inference::vad::machine::{
    SLOW_INFERENCE_THRESHOLD, VAD_WINDOW_SAMPLES, VadMachine, VadOptions, VadPredict, VadResult,
};

use super::Timeline;
use crate::sched::{Sampling, Sched, Step as Picked, Tick};

/// The scheduler's own random stream off the seed: the bytes `sched`.
const SCHED_STREAM: u64 = u64::from_be_bytes(*b"\0\0\0sched");

/// Who reports an event: the microphone, the model runtime, or the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Src {
    Mic,
    Model,
    Host,
}

impl crate::sched::Source for Src {}

/// What reaches the machine.
#[derive(Debug, Clone, PartialEq)]
pub enum Input {
    Frame { sample_rate: u32, channels: u16, samples_per_channel: u32 },
    Result(VadResult),
    Flush,
    Options(VadOptions),
}

/// What one delivery returned: the window handed out, the events, and a refused frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Returned {
    pub predict: Option<u64>,
    pub events: Vec<VadEvent>,
    pub refused: bool,
}

/// One delivered input and what it returned, at `at` ms.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub at: u64,
    pub input: Input,
    pub returned: Returned,
}

#[derive(Debug, Clone, Default)]
pub struct Trace {
    pub steps: Vec<Step>,
}

/// A finished run: what broke, and what it reached.
#[derive(Debug)]
pub struct Outcome {
    pub seed: u64,
    pub violations: Vec<String>,
    pub reached: BTreeSet<&'static str>,
    pub trace: Trace,
}

type Rule = (&'static str, fn(&Trace) -> Vec<String>);

pub const RULES: &[Rule] = &[
    ("vad_one_window_in_flight", one_in_flight),
    ("vad_stale_result_is_inert", stale_inert),
    ("vad_sample_index_advances_once_per_result", index_per_result),
    ("vad_speech_edges_alternate", edges_alternate),
];

/// Every rule over `trace`, each violation prefixed with its rule.
pub fn check(trace: &Trace) -> Vec<String> {
    RULES
        .iter()
        .flat_map(|(name, rule)| rule(trace).into_iter().map(move |v| format!("{name}: {v}")))
        .collect()
}

/// The person's speech as the model scores it: runs of speech and silence, each window's
/// probability drawn around its run's level, flickering across the thresholds.
struct Speaker {
    rng: Rng,
    speaking: bool,
    left: u32,
}

impl Speaker {
    fn probability(&mut self) -> f32 {
        if self.left == 0 {
            self.speaking = !self.speaking;
            self.left = match self.speaking {
                true => self.rng.u32(2..120),
                false => self.rng.u32(1..90),
            };
        }
        self.left -= 1;
        match (self.speaking, self.rng.u8(0..10)) {
            (true, 0) => self.rng.f32() * 0.5,
            (true, _) => 0.5 + self.rng.f32() * 0.5,
            (false, 0) => 0.3 + self.rng.f32() * 0.4,
            (false, _) => self.rng.f32() * 0.3,
        }
    }
}

/// The options a program sets: the defaults, or durations and thresholds of its own.
fn draw_options(rng: &mut Rng) -> VadOptions {
    if rng.bool() {
        return VadOptions::default();
    }
    let activation = 0.3 + rng.f64() * 0.6;
    VadOptions {
        min_speech_duration: rng.f64() * 0.3,
        min_silence_duration: rng.f64() * 1.0,
        prefix_padding_duration: rng.f64() * 1.0,
        max_buffered_speech: [0.5, 5.0, 60.0][rng.usize(0..3)],
        activation_threshold: activation,
        deactivation_threshold: (activation - rng.f64() * 0.3).max(0.01),
    }
}

/// One seeded run, the pending reports picked in the `sampling` order.
pub fn run(seed: u64, sampling: Sampling) -> Outcome {
    let mut rng = Rng::with_seed(seed);
    let mut sched: Sched<Src, Input> =
        Sched::new(sampling, Rng::with_seed(seed ^ SCHED_STREAM), &[]);
    let mut timeline: Timeline<Src, Input> = Timeline::default();
    let mut speaker = Speaker { rng: Rng::with_seed(rng.u64(..)), speaking: false, left: 0 };
    let contended = rng.u8(0..4) == 0;

    let sample_rate = [16_000, 24_000, 44_100, 48_000][rng.usize(0..4)];
    let channels = if rng.u8(0..4) == 0 { 2 } else { 1 };
    let frame_ms = if rng.bool() { 10 } else { 20 };
    let duration_ms = rng.u64(1_000..12_000);
    for at in (0..duration_ms).step_by(frame_ms as usize) {
        // a track switched under the stream: one frame at another rate
        let rate = if rng.u16(0..2000) == 0 { 8_000 } else { sample_rate };
        let samples_per_channel = rate * frame_ms / 1000;
        let frame = Input::Frame { sample_rate: rate, channels, samples_per_channel };
        timeline.at(at, Src::Mic, frame);
    }
    for _ in 0..rng.usize(0..4) {
        timeline.at(rng.u64(0..duration_ms), Src::Host, Input::Flush);
    }
    for _ in 0..rng.usize(0..3) {
        timeline.at(rng.u64(0..duration_ms), Src::Host, Input::Options(draw_options(&mut rng)));
    }

    let mut machine = VadMachine::new(draw_options(&mut rng));
    let mut trace = Trace::default();
    let mut now = 0;
    loop {
        timeline.release(now, &mut sched);
        let input = match sched.tick(now, &[], timeline.next_due()) {
            Tick::Step(Picked::Item { item, .. }) => item,
            Tick::Step(Picked::Process(_)) => unreachable!("no processes"),
            Tick::Advance(at) => {
                now = at;
                continue;
            }
            Tick::Quiet => break,
        };
        let returned = deliver(&mut machine, &input);
        if let Some(window) = returned.predict {
            let duration = match contended && rng.u8(0..8) == 0 {
                true => rng.u64(50..400),
                false => rng.u64(1..6),
            };
            let result = VadResult {
                window,
                probability: speaker.probability(),
                inference_duration: duration as f64 / 1000.0,
            };
            timeline.at(now + duration, Src::Model, Input::Result(result));
        }
        trace.steps.push(Step { at: now, input, returned });
    }
    let violations = check(&trace);
    Outcome { seed, violations, reached: reached(&trace), trace }
}

/// One input through the machine.
fn deliver(machine: &mut VadMachine, input: &Input) -> Returned {
    let predict = |p: Option<VadPredict>| {
        p.map(|p| {
            assert_eq!(p.samples.len(), VAD_WINDOW_SAMPLES, "a window is one model frame");
            p.window
        })
    };
    match input {
        Input::Frame { sample_rate, channels, samples_per_channel } => {
            let n = *channels as usize * *samples_per_channel as usize;
            // a tone under the noise floor: the content is the model's to score, not the frame's
            let data: Vec<u8> = (0..n).flat_map(|i| ((i % 64) as i16 * 16).to_le_bytes()).collect();
            let frame = AudioFrame::new(data, *sample_rate, *channels, *samples_per_channel);
            match machine.push(frame) {
                Ok(p) => Returned { predict: predict(p), ..Returned::default() },
                Err(_) => Returned { refused: true, ..Returned::default() },
            }
        }
        Input::Result(result) => {
            let (outputs, next) = machine.predicted(*result);
            let events = outputs.into_iter().map(|o| o.event).collect();
            Returned { predict: predict(next), events, refused: false }
        }
        Input::Flush => {
            machine.reset();
            Returned::default()
        }
        Input::Options(opts) => {
            machine.update_options(*opts);
            Returned::default()
        }
    }
}

/// What a trace reached, for the campaign's reach report.
pub fn reached(trace: &Trace) -> BTreeSet<&'static str> {
    let mut reached = BTreeSet::new();
    let mut in_flight = None;
    for step in &trace.steps {
        match &step.input {
            Input::Result(r) if in_flight != Some(r.window) => {
                reached.insert("VadStaleResult");
            }
            Input::Result(r) if r.inference_duration > SLOW_INFERENCE_THRESHOLD => {
                reached.insert("VadSlowInference");
            }
            Input::Flush if in_flight.is_some() => {
                reached.insert("VadFlushWithWindowInFlight");
            }
            Input::Options(_) => {
                reached.insert("VadOptionsUpdated");
            }
            _ => {}
        }
        if step.returned.refused {
            reached.insert("VadFrameRateRefused");
        }
        for event in &step.returned.events {
            reached.insert(match event.kind {
                VadEventType::StartOfSpeech => "VadStartOfSpeech",
                VadEventType::EndOfSpeech => "VadEndOfSpeech",
                VadEventType::InferenceDone => "VadInferenceDone",
            });
        }
        in_flight = window_after(in_flight, step);
    }
    reached
}

/// The window in flight after `step`, by what the trace shows: handed out, answered, or cut by
/// a flush.
fn window_after(in_flight: Option<u64>, step: &Step) -> Option<u64> {
    let mut now = match &step.input {
        Input::Flush => None,
        Input::Result(r) if in_flight == Some(r.window) => None,
        _ => in_flight,
    };
    if let Some(w) = step.returned.predict {
        now = Some(w);
    }
    now
}

fn one_in_flight(trace: &Trace) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_flight: Option<u64> = None;
    let mut last: Option<u64> = None;
    for step in &trace.steps {
        let freed = match &step.input {
            Input::Flush => true,
            Input::Result(r) => in_flight == Some(r.window),
            _ => false,
        };
        if let Some(w) = step.returned.predict {
            if in_flight.is_some() && !freed {
                out.push(format!(
                    "at {} ms window {w} handed out with {in_flight:?} in flight",
                    step.at
                ));
            }
            if last.is_some_and(|l| w <= l) {
                out.push(format!("at {} ms window {w} after window {last:?}", step.at));
            }
            last = Some(w);
        }
        in_flight = window_after(in_flight, step);
    }
    out
}

fn stale_inert(trace: &Trace) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_flight = None;
    for step in &trace.steps {
        if let Input::Result(r) = &step.input
            && in_flight != Some(r.window)
            && step.returned != Returned::default()
        {
            out.push(format!(
                "at {} ms the stale result for window {} returned {:?}",
                step.at, r.window, step.returned
            ));
        }
        in_flight = window_after(in_flight, step);
    }
    out
}

fn index_per_result(trace: &Trace) -> Vec<String> {
    let mut out = Vec::new();
    let mut accepted = 0u64;
    let mut in_flight = None;
    for step in &trace.steps {
        let fresh = matches!(&step.input, Input::Result(r) if in_flight == Some(r.window));
        if matches!(step.input, Input::Flush) {
            accepted = 0;
        }
        let done: Vec<&VadEvent> =
            step.returned.events.iter().filter(|e| e.kind == VadEventType::InferenceDone).collect();
        if fresh {
            accepted += 1;
            let index = accepted * VAD_WINDOW_SAMPLES as u64;
            if done.len() != 1 || done[0].samples_index != index {
                let got: Vec<u64> = done.iter().map(|e| e.samples_index).collect();
                out.push(format!(
                    "at {} ms result {accepted} since the flush gave {got:?}, not [{index}]",
                    step.at
                ));
            }
            for edge in
                step.returned.events.iter().filter(|e| e.kind != VadEventType::InferenceDone)
            {
                if edge.samples_index != index {
                    out.push(format!(
                        "at {} ms a {:?} at sample {} after window end {index}",
                        step.at, edge.kind, edge.samples_index
                    ));
                }
            }
        } else if !done.is_empty() {
            out.push(format!("at {} ms an inference_done without an accepted result", step.at));
        }
        in_flight = window_after(in_flight, step);
    }
    out
}

fn edges_alternate(trace: &Trace) -> Vec<String> {
    let mut out = Vec::new();
    let mut speaking = false;
    for step in &trace.steps {
        if matches!(step.input, Input::Flush) {
            speaking = false;
        }
        for e in &step.returned.events {
            match e.kind {
                VadEventType::StartOfSpeech if speaking => {
                    out.push(format!("at {} ms a second start_of_speech", step.at))
                }
                VadEventType::EndOfSpeech if !speaking => {
                    out.push(format!("at {} ms end_of_speech without a start", step.at))
                }
                VadEventType::StartOfSpeech | VadEventType::EndOfSpeech => speaking = !speaking,
                // reported before the window's own edge
                VadEventType::InferenceDone if e.speaking != speaking => out.push(format!(
                    "at {} ms inference_done says speaking={} after the edges left {speaking}",
                    step.at, e.speaking
                )),
                VadEventType::InferenceDone => {}
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
