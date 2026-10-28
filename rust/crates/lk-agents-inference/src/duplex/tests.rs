//! The reference's `tests/test_duplex_adapter.py`, over the segmenter and the gates.

use std::time::Duration;

use lk_agents_abi::llm::realtime::Event;
use lk_agents_abi::{AudioFrame, ChatContext, ChatItem, ChatMessage, FunctionCall, Role, Ts};

use super::gate::{AdaptiveNoiseGate, AudioGate, FixedGate};
use super::segmenter::{Fragment, Segmenter};

const SAMPLE_RATE: u32 = 24_000;
const FRAME_MS: u32 = 100;

/// A frame whose RMS is exactly `level`, from an alternating square wave.
fn frame_ms(level: f64, ms: u32) -> AudioFrame {
    let n = SAMPLE_RATE * ms / 1000;
    let amp = (level * 32767.0) as i16;
    let data: Vec<u8> = (0..n).flat_map(|i| if i % 2 == 0 { amp } else { -amp }.to_le_bytes()).collect();
    AudioFrame::new(data, SAMPLE_RATE, 1, n)
}

fn frame(level: f64) -> AudioFrame {
    frame_ms(level, FRAME_MS)
}

fn adaptive() -> AdaptiveNoiseGate {
    AdaptiveNoiseGate::default()
}

fn up(gate: &mut dyn AudioGate, level: f64) -> bool {
    gate.update(&frame(level)).unwrap()
}

// ------------------------------------------------------------------------------------ the gate

#[test]
fn gate_stays_closed_on_silence_and_steady_room_tone() {
    let mut g = adaptive();
    assert!(!(0..20).any(|_| up(&mut g, 0.0)));
    let mut g = adaptive();
    assert!(!(0..40).any(|_| up(&mut g, 0.02)), "a constant floor is never output");
}

#[test]
fn gate_opens_on_speech_over_room_tone() {
    let mut g = adaptive();
    for _ in 0..30 {
        up(&mut g, 0.02);
    }
    assert!(up(&mut g, 0.3));
}

#[test]
fn gate_holds_through_the_silence_window_then_closes() {
    let mut g = AdaptiveNoiseGate::new(3.0, 1.8, 0.25, 10.0);
    for _ in 0..30 {
        up(&mut g, 0.001);
    }
    assert!(up(&mut g, 0.3));
    assert!(up(&mut g, 0.001));
    assert!(up(&mut g, 0.001));
    assert!(!up(&mut g, 0.001));
}

#[test]
fn speech_longer_than_the_window_does_not_raise_the_floor_to_its_own_level() {
    let mut g = AdaptiveNoiseGate::new(3.0, 1.8, 0.5, 1.0);
    for _ in 0..10 {
        up(&mut g, 0.001);
    }
    assert!(up(&mut g, 0.3));
    for _ in 0..10 {
        for _ in 0..9 {
            assert!(up(&mut g, 0.3));
        }
        assert!(up(&mut g, 0.002));
    }
    assert!(up(&mut g, 0.3));
    for _ in 0..4 {
        assert!(up(&mut g, 0.001));
    }
    assert!(!up(&mut g, 0.001));
    assert!(up(&mut g, 0.3));
}

#[test]
fn a_gapless_monologue_never_wedges_the_gate_shut_and_one_dropped_frame_is_no_floor() {
    let mut g = adaptive();
    for _ in 0..30 {
        up(&mut g, 0.002);
    }
    assert!((0..400).all(|_| up(&mut g, 0.2)));
    for _ in 0..5 {
        up(&mut g, 0.002);
    }
    assert!(!up(&mut g, 0.002));

    let mut g = adaptive();
    for _ in 0..30 {
        up(&mut g, 0.002);
    }
    up(&mut g, 0.0);
    assert!(!(0..300).any(|_| up(&mut g, 0.002)));
}

#[test]
fn the_floor_recovers_once_the_model_is_really_silent() {
    let mut g = adaptive();
    assert!(!(0..30).any(|_| up(&mut g, 0.2)), "speech for a floor at first");
    for _ in 0..20 {
        up(&mut g, 0.002);
    }
    assert!(up(&mut g, 0.2));
}

#[test]
fn a_declared_silence_needs_no_learning_and_still_ignores_room_tone() {
    let mut g = FixedGate::new(0.002, 3.0, 1.8, 0.5);
    assert!(up(&mut g, 0.2));
    assert!((0..600).all(|_| up(&mut g, 0.2)));
    for _ in 0..5 {
        up(&mut g, 0.002);
    }
    assert!(!up(&mut g, 0.002));

    let mut g = FixedGate::new(0.02, 3.0, 1.8, 0.5);
    assert!(!(0..40).any(|_| up(&mut g, 0.02)));
    assert!(up(&mut g, 0.3));
}

#[test]
fn gate_decides_on_audio_duration_not_on_frame_count() {
    let levels: Vec<f64> = [vec![0.001; 12], vec![0.3; 6], vec![0.001; 12]].concat();
    let decisions: Vec<Vec<bool>> = [20u32, FRAME_MS]
        .iter()
        .map(|&ms| {
            let mut g = AdaptiveNoiseGate::new(3.0, 1.8, 0.45, 1.0);
            levels
                .iter()
                .map(|&l| {
                    let mut s = false;
                    for _ in 0..FRAME_MS / ms {
                        s = g.update(&frame_ms(l, ms)).unwrap();
                    }
                    s
                })
                .collect()
        })
        .collect();
    assert_eq!(decisions[0], decisions[1]);
    assert!(decisions[0][12] && !decisions[0][29]);
}

// ------------------------------------------------------------------------------- the segmenter

struct Fake {
    seg: Segmenter,
    now: Ts,
    events: Vec<Event>,
}

impl Fake {
    fn new() -> Self {
        let mut n = 0;
        let ids = Box::new(move || {
            n += 1;
            format!("item_{n}")
        });
        Fake { seg: Segmenter::new(Box::new(adaptive()), "fake", super::AUDIO_TIMEOUT, ids), now: Ts::ZERO, events: vec![] }
    }

    fn push(&mut self, level: f64, count: usize) {
        self.push_at(level, count, None);
    }

    fn push_at(&mut self, level: f64, count: usize, start_ms: Option<i64>) {
        for i in 0..count {
            let at = start_ms.map(|s| s + (i as i64) * i64::from(FRAME_MS));
            let evs = self.seg.on_audio(self.now, frame(level), at);
            self.events.extend(evs);
        }
    }

    fn say(&mut self, text: &str, start: i64, end: i64) {
        let evs = self.seg.on_transcript(Fragment { text: text.into(), start_ms: Some(start), end_ms: Some(end) });
        self.events.extend(evs);
    }

    fn end(&mut self) {
        let evs = self.seg.on_output_ended(self.now, None);
        self.events.extend(evs);
    }

    /// Per generation: (frames forwarded, transcript, finished).
    fn generations(&self) -> Vec<(String, usize, String, bool)> {
        let mut out: Vec<(String, usize, String, bool)> = vec![];
        for ev in &self.events {
            match ev {
                Event::GenerationCreated(g) => out.push((g.response_id.clone(), 0, String::new(), false)),
                Event::AudioDelta(a) => out.iter_mut().find(|g| g.0 == a.message_id).unwrap().1 += 1,
                Event::TextDelta(t) => out.iter_mut().find(|g| g.0 == t.message_id).unwrap().2.push_str(&t.delta),
                Event::GenerationDone(d) => out.iter_mut().find(|g| g.0 == d.response_id).unwrap().3 = true,
                _ => {}
            }
        }
        out
    }

    fn starts(&self, id: &str) -> Vec<Option<f64>> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Event::TextDelta(t) if t.message_id == id => Some(t.start_time),
                _ => None,
            })
            .collect()
    }
}

#[test]
fn silence_alone_produces_no_generation_and_an_untranscribed_burst_still_plays() {
    let mut f = Fake::new();
    f.push(0.0, 20);
    assert!(f.generations().is_empty());

    let mut f = Fake::new();
    f.push(0.001, 20);
    f.push(0.3, 3);
    f.push(0.001, 5);
    let g = f.generations();
    assert_eq!(g.len(), 1);
    assert!(g[0].1 >= 3 && g[0].2.is_empty());
}

#[test]
fn two_stretches_are_two_generations_and_a_pause_stays_in_one() {
    let mut f = Fake::new();
    f.push(0.001, 20);
    f.push(0.3, 3);
    f.push(0.001, 8);
    f.push(0.3, 3);
    f.push(0.001, 8);
    assert_eq!(f.generations().len(), 2);

    let mut f = Fake::new();
    f.push(0.001, 20);
    f.push(0.3, 3);
    f.push(0.001, 3);
    f.push(0.3, 3);
    f.push(0.001, 8);
    let g = f.generations();
    assert_eq!(g.len(), 1);
    assert!(g[0].1 >= 9, "both stretches and the pause between them");
}

#[test]
fn fragments_join_the_open_burst_timed_against_it() {
    let mut f = Fake::new();
    f.push(0.001, 20);
    f.push(0.3, 2);
    f.say("hello", 7000, 7200);
    f.say(" there", 7200, 7400);
    f.push(0.3, 3);
    f.end();
    let g = f.generations();
    assert_eq!(g[0].2, "hello there");
    assert_eq!(f.starts(&g[0].0), vec![Some(0.0), Some(0.2)]);
}

#[test]
fn stamped_audio_supplies_the_clock() {
    let mut f = Fake::new();
    f.push_at(0.001, 20, Some(5000));
    f.push_at(0.3, 2, Some(7000));
    assert_eq!(f.seg.audio_ms(), 7200);
    f.say("hello", 7000, 7200);
    f.push_at(0.3, 3, Some(7200));
    f.end();
    let id = f.generations()[0].0.clone();
    assert_eq!(f.starts(&id), vec![Some(0.0)]);
}

#[test]
fn a_fragment_that_leads_its_audio_waits_for_the_burst_that_carries_it() {
    let mut f = Fake::new();
    f.push(0.001, 20);
    f.say(" Sure.", 2000, 2200);
    f.push(0.001, 5);
    assert!(f.generations().is_empty());
    assert_eq!(f.seg.waiting_fragments(), 1);
    f.push(0.3, 3);
    f.push(0.001, 8);
    let g = f.generations();
    assert_eq!((g.len(), g[0].1, g[0].2.as_str()), (1, 7, " Sure."));
}

#[test]
fn a_fragment_the_burst_ends_before_reaching_rolls_over_to_the_next() {
    let mut f = Fake::new();
    f.push(0.001, 20);
    f.push(0.3, 3);
    f.say(" Checking the current conditions.", 2000, 2800);
    f.say(" It's", 4200, 4400);
    f.push(0.001, 8);
    assert_eq!(f.generations().len(), 1);
    assert_eq!(f.seg.waiting_fragments(), 1);
    assert_eq!(f.generations()[0].2, " Checking the current conditions.");
    f.push(0.3, 3);
    f.say(" 62 degrees.", 4400, 5000);
    f.push(0.3, 3);
    f.push(0.001, 8);
    let g = f.generations();
    assert_eq!(g.len(), 2);
    assert_eq!(g[1].2, " It's 62 degrees.");
}

#[test]
fn a_fragment_is_attached_when_the_audio_reaches_it_within_the_lead() {
    let mut f = Fake::new();
    f.push(0.001, 20);
    f.push(0.3, 3);
    f.say(" Yep, I've got order A1042", 5000, 5600);
    f.say(" on file.", 6500, 6900);
    f.push(0.3, 1);
    assert_eq!(f.seg.waiting_fragments(), 1, "due 1.5 s after the onset; the audio is at 0.4 s");
    f.push(0.3, 2);
    f.push(0.001, 4);
    f.push(0.3, 3);
    assert_eq!(f.seg.waiting_fragments(), 0);
    f.push(0.001, 8);
    let g = f.generations();
    assert_eq!(g.len(), 1);
    assert_eq!(g[0].2, " Yep, I've got order A1042 on file.");
}

#[test]
fn words_after_the_last_frame_join_their_burst_and_late_words_go_out_alone() {
    let mut f = Fake::new();
    f.push(0.001, 20);
    f.push(0.3, 3);
    f.say("Hello there.", 5000, 5400);
    f.end();
    assert_eq!(f.generations()[0].2, "Hello there.");

    let mut f = Fake::new();
    f.push(0.001, 20);
    f.push(0.3, 3);
    f.say("Hello", 2000, 2200);
    f.push(0.001, 8);
    assert_eq!(f.generations().len(), 1);
    f.say(" there.", 2200, 2300);
    let g = f.generations();
    assert_eq!(g.len(), 2);
    assert_eq!((g[1].1, g[1].2.as_str()), (0, " there."));
    f.say("Next.", 5000, 5200);
    f.push(0.3, 3);
    f.push(0.001, 8);
    let g = f.generations();
    assert_eq!(g.len(), 3);
    assert_eq!(f.starts(&g[2].0), vec![Some(0.0)], "the late words anchored nothing");
}

// Catches: words no sound claims dropped, or emitted before 3 s of the model's own silence.
#[test]
fn transcript_no_audio_ever_claims_is_emitted_after_three_seconds() {
    let mut f = Fake::new();
    f.push(0.001, 20);
    f.say("Lost words.", 2000, 2400);
    f.push(0.001, 29);
    assert!(f.generations().is_empty(), "2.9 s is not yet the timeout");
    f.push(0.001, 1);
    let g = f.generations();
    assert_eq!((g.len(), g[0].1, g[0].2.as_str(), g[0].3), (1, 0, "Lost words.", true));
    assert_eq!(f.seg.waiting_fragments(), 0);
}

#[allow(clippy::disallowed_methods, reason = "S0: existing test harness live resources")]
fn call() -> FunctionCall {
    FunctionCall {
        id: "call_1".into(),
        call_id: "c1".into(),
        arguments: "{}".into(),
        name: "lookup".into(),
        created_at: std::time::SystemTime::now(),
        extra: Default::default(),
        group_id: None,
    }
}

#[test]
fn a_function_call_joins_the_speech_in_flight_or_is_a_generation_of_its_own() {
    let mut f = Fake::new();
    f.push(0.001, 20);
    f.push(0.3, 3);
    let evs = f.seg.on_function_call(call());
    assert!(matches!(evs.as_slice(), [Event::FunctionCall(_)]));
    assert!(f.seg.burst_open());

    let mut f = Fake::new();
    f.push(0.001, 20);
    let kinds: Vec<String> = f
        .seg
        .on_function_call(call())
        .iter()
        .map(|e| format!("{e:?}").split('(').next().unwrap().to_owned())
        .collect();
    assert_eq!(kinds, ["GenerationCreated", "FunctionCall", "GenerationDone"]);
    assert!(!f.seg.burst_open());
}

#[test]
fn reconnect_releases_the_burst_and_the_words_waiting_on_it() {
    let mut f = Fake::new();
    f.push(0.001, 20);
    f.push(0.3, 3);
    f.say("Half a", 2000, 2200);
    f.say(" sentence", 4000, 4400);
    f.push(0.3, 1);
    assert_eq!(f.seg.waiting_fragments(), 1);
    let evs = f.seg.on_reconnected();
    f.events.extend(evs);
    assert!(!f.seg.burst_open() && f.seg.waiting_fragments() == 0);
    f.push(0.001, 3);
    let g = f.generations();
    assert_eq!(g.len(), 1);
    assert_eq!((g[0].1, g[0].2.as_str()), (4, "Half a"));
    assert!(matches!(f.events.last(), Some(Event::SessionReconnected(_))));
}

// Catches: the stream that stops mid-burst never ending its generation (0.8 s plus the frame in
// hand), and the gate kept open across the gap.
#[test]
fn a_stream_that_stops_mid_burst_ends_the_generation_on_the_audio_timeout() {
    let mut f = Fake::new();
    f.push(0.001, 20);
    f.push(0.3, 3);
    assert!(f.seg.burst_open());
    let deadline = f.seg.next_deadline().expect("armed");
    assert_eq!(deadline, Ts::ZERO.saturating_add(Duration::from_millis(900)));
    let _ = f.seg.tick(Ts::ZERO.saturating_add(Duration::from_millis(899)));
    assert!(f.seg.burst_open());
    let evs = f.seg.tick(deadline);
    f.events.extend(evs);
    assert!(!f.seg.burst_open());
    f.push(0.3, 3);
    assert_eq!(f.generations().len(), 2);
}

// Catches: a reply answered by a lone call or by late words, never failed when superseded,
// reconnected or timed out (10 s), or answered as unsolicited speech.
#[test]
fn a_requested_reply_is_the_speech_that_follows_it_and_otherwise_fails() {
    let mut f = Fake::new();
    assert!(f.seg.generate_reply(Ts::ZERO, "r1".into()).is_empty());
    f.push(0.001, 20);
    let lone = f.seg.on_function_call(call());
    assert!(matches!(&lone[0], Event::GenerationCreated(g) if g.req_id.is_none()));
    f.push(0.5, 3);
    let speech = f.events.iter().find_map(|e| match e {
        Event::GenerationCreated(g) => Some(g.req_id.clone()),
        _ => None,
    });
    assert_eq!(speech, Some(Some("r1".to_owned())));

    let mut f = Fake::new();
    let _ = f.seg.generate_reply(Ts::ZERO, "r1".into());
    let evs = f.seg.generate_reply(Ts::ZERO, "r2".into());
    assert!(matches!(&evs[..], [Event::GenerateReplyFailed(x)] if x.req_id == "r1" && x.error.message.contains("superseded")));
    let evs = f.seg.on_reconnected();
    assert!(matches!(&evs[0], Event::GenerateReplyFailed(x) if x.error.message.contains("reconnected")));
    let _ = f.seg.generate_reply(Ts::ZERO, "r3".into());
    assert!(f.seg.tick(Ts::ZERO.saturating_add(Duration::from_millis(9_999))).is_empty());
    let evs = f.seg.tick(Ts::ZERO.saturating_add(Duration::from_secs(10)));
    assert!(matches!(&evs[..], [Event::GenerateReplyFailed(x)] if x.req_id == "r3"));
    let _ = f.seg.generate_reply(Ts::ZERO, "r4".into());
    let evs = f.seg.close();
    assert!(matches!(&evs[..], [Event::GenerateReplyFailed(x)] if x.error.message.contains("closed")));
}

#[allow(clippy::disallowed_methods, reason = "S0: existing test harness live resources")]
#[test]
fn the_adapter_keeps_the_context_under_the_frameworks_ids_and_sends_only_what_is_new() {
    let mut f = Fake::new();
    f.seg.on_input_transcription("u1", "What is the weather?", true, None, std::time::SystemTime::now());
    f.push(0.001, 20);
    f.push(0.3, 3);
    f.say("Let me check.", 2000, 2400);
    f.push(0.3, 1);
    let _ = f.seg.on_function_call(call());
    f.push(0.001, 8);
    let items = f.seg.chat_ctx().items.clone();
    let ids: Vec<&str> = items.iter().map(ChatItem::id).collect();
    let gen_id = f.generations()[0].0.clone();
    assert_eq!(ids, ["u1", gen_id.as_str(), "call_1"]);

    let mut ctx = f.seg.chat_ctx().clone();
    ctx.insert(ChatItem::Message(ChatMessage::text("typed_1", Role::User, "typed by the app", std::time::SystemTime::now())));
    let new = f.seg.update_chat_ctx(&ctx);
    assert_eq!(new.iter().map(ChatItem::id).collect::<Vec<_>>(), ["typed_1"]);

    let mut ctx = f.seg.chat_ctx().clone();
    ctx.items.retain(|i| i.id() != "u1");
    assert!(f.seg.update_chat_ctx(&ctx).is_empty(), "a removal is logged, never sent");
}

#[allow(clippy::disallowed_methods, reason = "S0: existing test harness live resources")]
#[test]
fn startup_history_seeds_both_the_adapter_and_the_model() {
    let mut f = Fake::new();
    let ctx = ChatContext::new(vec![ChatItem::Message(ChatMessage::text("m1", Role::User, "a prior turn", std::time::SystemTime::now()))]);
    let items = f.seg.seed(&ctx);
    assert_eq!(items.iter().map(ChatItem::id).collect::<Vec<_>>(), ["m1"]);
    assert_eq!(f.seg.chat_ctx().items.len(), 1);
}

struct Failing;

impl AudioGate for Failing {
    fn update(&mut self, _frame: &AudioFrame) -> Result<bool, String> {
        Err("Boom".into())
    }
    fn deactivate(&mut self) {}
}

#[test]
fn a_failed_gate_reports_an_unrecoverable_error_under_the_models_label() {
    let mut seg = Segmenter::new(Box::new(Failing), "plugin.Model", super::AUDIO_TIMEOUT, Box::new(|| "item_1".into()));
    let evs = seg.on_audio(Ts::ZERO, frame(0.5), None);
    assert!(matches!(&evs[..], [Event::Error(e)] if !e.recoverable && e.label == "plugin.Model"));
    assert!(seg.on_audio(Ts::ZERO, frame(0.5), None).is_empty(), "the output is over");
}
