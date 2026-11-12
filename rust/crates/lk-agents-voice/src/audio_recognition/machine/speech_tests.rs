use super::super::{AudioRecognition, EndOfTurnInfo, WallAnchor};
use super::*;
use crate::agent_session::options::ConfigStack;
use crate::audio_recognition::{UserTurnCommitDecision, UserTurnCommitVerdict};
use lk_agents_abi::stt::events::SpeechData;
use lk_agents_abi::voice::session_options::SessionLegs;

fn ms(n: u64) -> Ts {
    Ts::from_nanos(n * 1_000_000)
}

/// One recogniser in STT turn detection with no VAD, driven the way the session driver does:
/// deadlines fire when time passes them, and every committed turn gets a replying verdict.
struct Turns {
    r: AudioRecognition,
    now: Ts,
    deadlines: Vec<(Timer, Ts)>,
    ended: Vec<EndOfTurnInfo>,
}

impl Turns {
    /// The STT stream's first frame was heard at `input_started_at`.
    fn stt_only(input_started_at: Ts) -> Turns {
        let legs = SessionLegs { stt: true, ..SessionLegs::none() };
        let mut r = AudioRecognition::new(&ConfigStack::default().resolve(&legs), &legs);
        let eff = r.set_turn_detection_mode(Ts::ZERO, TurnDetectionMode::Stt);
        r.input_started_at = Some(input_started_at);
        r.wall_anchor = Some(WallAnchor { wall: 1_000.0, at: Ts::ZERO });
        let mut turns = Turns { r, now: Ts::ZERO, deadlines: Vec::new(), ended: Vec::new() };
        turns.apply(eff);
        turns
    }

    fn apply(&mut self, eff: Vec<AudioRecognitionEffect>) {
        for e in eff {
            match e {
                AudioRecognitionEffect::SetDeadline(timer, at) => {
                    self.deadlines.retain(|(t, _)| *t != timer);
                    self.deadlines.push((timer, at));
                }
                AudioRecognitionEffect::ClearDeadline(timer) => {
                    self.deadlines.retain(|(t, _)| *t != timer);
                }
                AudioRecognitionEffect::EndOfTurn(info) => {
                    let verdict = if info.late {
                        UserTurnCommitVerdict::LateTranscript
                    } else {
                        UserTurnCommitVerdict::Reply
                    };
                    let decision = UserTurnCommitDecision {
                        verdict,
                        committed: true,
                        cancel_false_interruption: false,
                    };
                    let eff = self.r.on_commit_verdict(self.now, &info, &decision);
                    self.ended.push(info);
                    self.apply(eff);
                }
                _ => {}
            }
        }
    }

    /// Moves time to `at`, firing every deadline due on the way.
    fn at(&mut self, at: Ts) {
        while let Some(&(timer, due)) =
            self.deadlines.iter().filter(|(_, due)| *due <= at).min_by_key(|(_, due)| *due)
        {
            self.deadlines.retain(|(t, _)| *t != timer);
            self.now = due;
            let eff = self.r.on_timer(due, timer);
            self.apply(eff);
        }
        self.now = at;
    }

    fn stt(&mut self, at: Ts, ev: SpeechEvent) {
        self.at(at);
        let eff = self.r.on_stt_event(at, &ev);
        self.apply(eff);
    }

    fn start(&mut self, at: Ts, onset_wall: Option<f64>) {
        let mut ev = SpeechEvent::new(SpeechEventType::StartOfSpeech);
        ev.speech_start_time = onset_wall;
        self.stt(at, ev);
    }

    /// A final whose last word ends `word_end` seconds into the stream.
    fn final_(&mut self, at: Ts, text: &str, word_end: f64) {
        let mut ev = SpeechEvent::new(SpeechEventType::FinalTranscript);
        let mut alt = SpeechData::new("en", text);
        alt.end_time = word_end;
        ev.alternatives.push(alt);
        self.stt(at, ev);
    }

    /// The provider's end of speech, with the last word's end when it sends one.
    fn end(&mut self, at: Ts, word_end: Option<f64>) {
        let mut ev = SpeechEvent::new(SpeechEventType::EndOfSpeech);
        if let Some(word_end) = word_end {
            let mut alt = SpeechData::new("en", "");
            alt.end_time = word_end;
            ev.alternatives.push(alt);
        }
        self.stt(at, ev);
    }

    fn committed(&self) -> Vec<&str> {
        self.ended.iter().map(|i| i.new_transcript.as_str()).collect()
    }
}

/// On an STT stream opened mid-session, word times are anchored at that stream's first frame.
/// https://github.com/livekit/agents/issues/7651 https://github.com/livekit/agents/pull/7795
/// https://github.com/livekit/agents/pull/6207 https://github.com/livekit/agents/pull/5479
#[test]
fn stream_changes_preserve_session_audio_anchors() {
    // The reopened stream's first frame is 10 s into the session (wall 1010.0).
    let mut t = Turns::stt_only(ms(10_000));
    let min_delay = t.r.min_delay();
    t.start(ms(10_400), Some(1_010.2));
    t.final_(ms(11_300), "book a table", 1.0);
    t.end(ms(11_600), Some(1.0));
    t.at(ms(11_600).saturating_add(min_delay));

    let [turn] = t.ended.as_slice() else { panic!("{:?}", t.committed()) };
    assert_eq!(turn.new_transcript, "book a table");
    let m = &turn.metrics;
    assert_eq!(m.started_speaking_at, Some(ms(10_200)), "the server onset, session-relative");
    assert_eq!(m.stopped_speaking_at, Some(ms(11_000)), "the word's end on this stream");
    assert_eq!(m.transcription_delay, Some(Duration::from_millis(300)));
    assert_eq!(
        m.end_of_turn_delay,
        Some(min_delay.max(Duration::from_millis(600))),
        "the countdown ran from the word's end"
    );
}

/// Each turn's metrics use its own onset and last word's end, also with an untimed end of speech.
/// https://github.com/livekit/agents/issues/5876 https://github.com/livekit/agents/issues/7651
/// https://github.com/livekit/agents/issues/6093 https://github.com/livekit/agents/issues/2606
/// https://github.com/livekit/agents/pull/4396 https://github.com/livekit/agents/issues/5669
#[test]
#[ignore = "https://github.com/livekit/agents/issues/5876 reproduces: an untimestamped END_OF_SPEECH overwrites the final's word end, so stopped_speaking_at is its arrival and transcription_delay 0"]
fn successive_turn_metrics_use_fresh_valid_anchors() {
    let mut t = Turns::stt_only(Ts::ZERO);
    let min_delay = t.r.min_delay();
    // Turn one: speech 0.2..1.0 s, final at 1.3 s, end of speech (no timestamp) at 1.6 s.
    t.start(ms(200), None);
    t.final_(ms(1_300), "first", 1.0);
    t.end(ms(1_600), None);
    t.at(ms(1_600).saturating_add(min_delay));
    // Turn two: speech 5.0..6.0 s, final at 6.2 s, end of speech at 6.7 s.
    t.start(ms(5_000), None);
    t.final_(ms(6_200), "second", 6.0);
    t.end(ms(6_700), None);
    t.at(ms(6_700).saturating_add(min_delay));

    assert_eq!(t.committed(), ["first", "second"]);
    let starts: Vec<_> = t.ended.iter().map(|i| i.metrics.started_speaking_at).collect();
    assert_eq!(starts, [Some(ms(200)), Some(ms(5_000))], "each turn's own onset");
    let stops: Vec<_> = t.ended.iter().map(|i| i.metrics.stopped_speaking_at).collect();
    assert_eq!(stops, [Some(ms(1_000)), Some(ms(6_000))], "each turn's last word, not its EOS");
    let delays: Vec<_> = t.ended.iter().map(|i| i.metrics.transcription_delay).collect();
    assert_eq!(delays, [Some(Duration::from_millis(300)), Some(Duration::from_millis(200))]);
}

/// A final that arrives after the caller resumed commits with its own utterance, not the next.
/// https://github.com/livekit/agents/issues/7699 https://github.com/livekit/agents/issues/2642
/// https://github.com/livekit/agents/issues/3382 https://github.com/livekit/agents/issues/6504
#[test]
#[ignore = "https://github.com/livekit/agents/issues/7699 reproduces: a final landing after the commit of a later END_OF_SPEECH waits for the next utterance (fixed upstream in PR 7739)"]
fn late_finals_belong_to_the_correct_turn() {
    let mut t = Turns::stt_only(Ts::ZERO);
    let min_delay = t.r.min_delay();
    t.start(ms(100), None);
    t.end(ms(900), Some(0.8));
    // The caller resumes before A's final arrives.
    t.start(ms(1_000), None);
    t.final_(ms(1_100), "utterance a", 0.8);
    t.end(ms(2_000), Some(1.9));
    t.at(ms(2_000).saturating_add(min_delay));
    t.final_(ms(2_100).saturating_add(min_delay), "utterance b", 1.9);
    t.at(ms(5_000));
    assert_eq!(t.committed(), ["utterance a", "utterance b"], "B commits without waiting for C");

    t.start(ms(6_000), None);
    t.final_(ms(6_900), "utterance c", 6.8);
    t.end(ms(7_000), Some(6.8));
    t.at(ms(9_000));
    assert_eq!(t.committed(), ["utterance a", "utterance b", "utterance c"]);
}
