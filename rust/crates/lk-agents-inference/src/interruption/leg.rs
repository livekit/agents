//! The session loop's [`InterruptionPort`], satisfied by this module's detector: audio, sentinels
//! and verdicts stay in Rust. One stream per session, opened from the session's `open` (it spawns).

use std::sync::Arc;

use lk_agents_abi::turn::detector::LegEvent;
use lk_agents_abi::turn::interruption::{InterruptionPort, InterruptionSignal, OverlapVerdict};
use lk_agents_abi::{AudioFrame, StreamSink, WallAnchor};

use super::{
    AdaptiveInterruptionDetector, InputEvent, InterruptionStream, OverlapEnd, OverlapStart,
    OverlappingSpeechEvent, StreamEvent,
};

pub struct InterruptionLeg {
    /// The input half, cloned out of the stream: synchronous and unbounded, since `signal` may not
    /// block.
    input: tokio::sync::mpsc::UnboundedSender<InputEvent>,
    /// Held across an await by `next_event`, hence a `tokio` mutex.
    stream: tokio::sync::Mutex<InterruptionStream>,
    /// Converts sentinel `Ts`s to the gateway's `time.time()` seconds, on the session's anchor.
    anchor: WallAnchor,
}

impl InterruptionLeg {
    /// **Must be called on the tokio runtime.**
    pub fn open(detector: &AdaptiveInterruptionDetector, anchor: WallAnchor) -> Arc<Self> {
        let stream = detector.stream();
        // Taken before the stream is parked behind the mutex, so pushes never wait on verdicts.
        let input = stream.sender().expect("a fresh stream still has its input half");
        Arc::new(InterruptionLeg { input, stream: tokio::sync::Mutex::new(stream), anchor })
    }
}

/// The detector as the session's source of interruption legs: one per session, opened at `start()`.
impl lk_agents_abi::turn::interruption::InterruptionSource for AdaptiveInterruptionDetector {
    fn open(&self, anchor: WallAnchor) -> Arc<dyn InterruptionPort> {
        InterruptionLeg::open(self, anchor)
    }
}

#[async_trait::async_trait]
impl InterruptionPort for InterruptionLeg {
    fn audio(&self) -> Arc<dyn StreamSink<AudioFrame>> {
        Arc::new(InterruptionAudioSink(self.input.clone()))
    }

    fn signal(&self, signal: InterruptionSignal) {
        // A failed send means the task ended; the failure already reached the host's event
        // channel.
        let _ = self.input.send(sentinel(signal, self.anchor));
    }

    /// Verdicts and failures on one leg: a detector error reaches the loop as
    /// `LegEvent::Error { recoverable }` and is journalled (I-24).
    async fn next_event(&self) -> LegEvent<OverlapVerdict> {
        let mut stream = self.stream.lock().await;
        loop {
            return match stream.recv_event().await {
                Some(StreamEvent::Overlap(ev)) => {
                    LegEvent::Verdict { req: None, verdict: verdict(&ev, self.anchor) }
                }
                Some(StreamEvent::Error(e)) => LegEvent::Error {
                    req: None,
                    recoverable: e.recoverable,
                    err: lk_agents_abi::errors::ProviderError::Local(e.error),
                },
                // The loop pushes no span (see `sentinel`), so none comes back.
                Some(StreamEvent::SpanUpdate { .. }) => continue,
                None => LegEvent::Closed,
            };
        }
    }
}

fn sentinel(signal: InterruptionSignal, anchor: WallAnchor) -> InputEvent {
    match signal {
        InterruptionSignal::AgentSpeechStarted => InputEvent::AgentSpeechStarted,
        InterruptionSignal::AgentSpeechEnded => InputEvent::AgentSpeechEnded,
        InterruptionSignal::OverlapStarted { started_at, speech_duration } => {
            InputEvent::OverlapSpeechStarted(OverlapStart {
                speech_duration: speech_duration.as_secs_f64(),
                started_at: anchor.seconds(started_at),
                // No span: the loop holds none, so no `span_update` is emitted.
                user_speaking_span: None,
            })
        }
        InterruptionSignal::OverlapEnded { ended_at, agent_ended } => {
            InputEvent::OverlapSpeechEnded(OverlapEnd {
                ended_at: anchor.seconds(ended_at),
                agent_ended,
            })
        }
    }
}

fn verdict(ev: &OverlappingSpeechEvent, anchor: WallAnchor) -> OverlapVerdict {
    OverlapVerdict {
        is_interruption: ev.is_interruption,
        agent_ended: ev.agent_ended,
        // `ev.overlap_started_at or ev.detected_at`: Python's `or`, so `0.0` falls through.
        ignore_user_transcript_until: anchor
            .ts(ev.overlap_started_at.filter(|v| *v != 0.0).unwrap_or(ev.detected_at)),
    }
}

/// The detector's audio leg, as the loop's fan-out sees it. `AudioWindow::push_frame` resamples to
/// 16 kHz itself.
struct InterruptionAudioSink(tokio::sync::mpsc::UnboundedSender<InputEvent>);

impl StreamSink<AudioFrame> for InterruptionAudioSink {
    fn push(&self, frame: AudioFrame) -> Result<(), lk_agents_abi::errors::ProviderError> {
        // `Ok(())` even when the send fails: a dead detector falls back to VAD-only barge-in.
        let _ = self.0.send(InputEvent::Audio(frame));
        Ok(())
    }

    fn flush(&self) -> Result<(), lk_agents_abi::errors::ProviderError> {
        // `_FlushSentinel`, forwarded so both channels carry the same sequence.
        let _ = self.0.send(InputEvent::Flush);
        Ok(())
    }

    fn end_input(&self) {
        // Nothing: the detector's stream outlives a handoff.
    }
}

/// The port's two mappings, the part that can be asserted on without a socket.
#[cfg(test)]
mod tests {
    use std::time::Duration;

    use lk_agents_abi::turn::interruption::InterruptionSignal as S;
    use lk_agents_abi::{Ts, WallAnchor};

    use super::{InputEvent, OverlappingSpeechEvent, sentinel, verdict};

    fn anchor() -> WallAnchor {
        WallAnchor { wall: 1_787_000_000.0, at: Ts::from_nanos(10_000_000_000) }
    }

    fn event(started: Option<f64>, detected: f64) -> OverlappingSpeechEvent {
        OverlappingSpeechEvent {
            created_at: detected,
            detected_at: detected,
            is_interruption: true,
            agent_ended: false,
            total_duration: 0.4,
            prediction_duration: 0.2,
            detection_delay: 0.1,
            overlap_started_at: started,
            speech_input: None,
            probabilities: None,
            probability: 0.9,
            num_requests: 1,
        }
    }

    // Catches: an overlap sentinel carrying a monotonic `Ts` instead of wall seconds.
    #[test]
    fn the_two_overlap_sentinels_carry_wall_seconds_not_monotonic_ones() {
        let at = Ts::from_nanos(12_500_000_000); // 2.5 s after the anchor
        match sentinel(
            S::OverlapStarted { started_at: at, speech_duration: Duration::from_millis(300) },
            anchor(),
        ) {
            InputEvent::OverlapSpeechStarted(start) => {
                assert_eq!(start.started_at, 1_787_000_002.5);
                assert_eq!(start.speech_duration, 0.3);
                assert!(start.user_speaking_span.is_none(), "the loop holds no spans");
            }
            other => panic!("expected an overlap start, got {other:?}"),
        }

        match sentinel(S::OverlapEnded { ended_at: at, agent_ended: true }, anchor()) {
            InputEvent::OverlapSpeechEnded(end) => {
                assert_eq!(end.ended_at, 1_787_000_002.5);
                // `agent_ended` marks a force-ended overlap; lost, every agent finish counts as a
                // backchannel.
                assert!(end.agent_ended);
            }
            other => panic!("expected an overlap end, got {other:?}"),
        }
    }

    // Catches: the two agent-speech sentinels being swapped or collapsed.
    #[test]
    fn the_agent_speech_sentinels_map_one_to_one() {
        assert_eq!(sentinel(S::AgentSpeechStarted, anchor()), InputEvent::AgentSpeechStarted);
        assert_eq!(sentinel(S::AgentSpeechEnded, anchor()), InputEvent::AgentSpeechEnded);
    }

    // Catches: `ev.overlap_started_at or ev.detected_at` read as `unwrap_or`.
    #[test]
    fn a_zero_overlap_start_falls_through_to_the_detection_time() {
        let a = anchor();
        assert_eq!(
            verdict(&event(Some(1_787_000_001.0), 1_787_000_003.0), a).ignore_user_transcript_until,
            Ts::from_nanos(11_000_000_000)
        );
        for absent in [Some(0.0), None] {
            assert_eq!(
                verdict(&event(absent, 1_787_000_003.0), a).ignore_user_transcript_until,
                Ts::from_nanos(13_000_000_000),
                "a falsy overlap start is the reference's `or`, not an `unwrap_or`"
            );
        }
    }

    // Catches: a verdict's `is_interruption` and `agent_ended` folded into one.
    #[test]
    fn the_verdicts_two_flags_stay_apart() {
        let mut ev = event(Some(1_787_000_001.0), 1_787_000_003.0);
        ev.is_interruption = false;
        ev.agent_ended = true;
        let v = verdict(&ev, anchor());
        assert!(!v.is_interruption);
        assert!(v.agent_ended);
    }
}
