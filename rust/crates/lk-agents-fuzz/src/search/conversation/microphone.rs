//! Microphone segments and the detector verdicts delivered around them.

use super::*;
use crate::fakes::realtime::server::{
    SPEECH_STARTED_LAG_SEMANTIC, SPEECH_STARTED_LAG_SERVER, SPEECH_STOPPED_LAG_SEMANTIC,
    SPEECH_STOPPED_LAG_SERVER,
};

/// The person's side: a stretch of microphone, and what the providers say about it.
impl Talk {
    /// "Mm-hm" over the agent: short enough to be a backchannel, not a interruption.
    pub(super) fn backchannel(&mut self) -> Segment {
        let text = BACKCHANNELS[self.rng.usize(0..BACKCHANNELS.len())];
        let dur_ms = self.rng.u64(BACKCHANNEL_SAID_MS);
        Segment { speaking: true, dur_ms, pacing: Pacing::Steady, text: Some(text), done: false }
    }

    pub(super) fn segment(&mut self, speaking: bool, done: bool) -> Segment {
        let Persona { pace, silences, .. } = self.persona;
        let rng = &mut self.rng;
        // What the person says decides how long they speak: their words at their pace.
        let text = speaking.then(|| USER_TURNS[rng.usize(0..USER_TURNS.len())]);
        let dur_ms = match text {
            Some(text) => {
                let words = text.split_whitespace().count() as u64;
                (words * 10_000 * 100 / (rng.u64(WORDS_PER_SEC_X10) * pace)).max(MIN_SPEECH_MS)
            }
            None => rng.u64(SILENCE_MS) * silences / 100,
        };
        let pacing = match rng.u8(0..3) {
            0 => Pacing::Steady,
            1 => Pacing::Jitter { spread_ms: rng.u64(JITTER_MS) },
            _ => Pacing::Gap { stall_ms: rng.u64(STALL_MS) },
        };
        Segment { speaking, dur_ms, pacing, text, done }
    }

    /// What the detector and the recogniser will say about this speech, in due order.
    ///
    /// The recogniser's end of speech is placed before or after the detector's with equal
    /// probability. That race is the point: the turn machine has two owners for the same edge
    /// and the journal says which one it believed.
    pub(super) fn verdicts(&mut self, speech: &Segment) -> Vec<Verdict> {
        let live = self.run.record.live();
        let vad = !live.of(Kind::Vad).is_empty();
        let noise = self.persona.noise;
        let Providers { latency, late_finals, revisions, .. } = self.providers;
        let rng = &mut self.rng;
        let slow = |ms: u64| ms * latency / 100;
        // Noise: the detector hears speech, the recogniser has nothing to transcribe.
        let stt = !live.of(Kind::Stt).is_empty() && rng.u8(0..100) >= noise;
        let dur = speech.dur_ms;
        let mut out = Vec::new();
        if vad {
            out.push(Verdict { at_ms: slow(rng.u64(VAD_START_MS)), word: Word::VadStart });
        }
        let vad_end_at = dur + slow(rng.u64(VAD_END_MS));
        // The recogniser behaves as the measured model the setup names does: when its first words
        // come, how often it restates them, how it cuts a long turn into finals, what it sends
        // around each final (a usage report before it, the end of speech with it), and its
        // quirks: Deepgram's empty finals over noise, AssemblyAI's second final for the same
        // words, Flux's preflights.
        let model = &STT_MODELS[self.run.setup.stt_model % STT_MODELS.len()];
        let timing = self.run.setup.timing();
        let listening = !live.of(Kind::Stt).is_empty();
        if let (true, Some(text)) = (stt, speech.text) {
            let knobs = SttKnobs { latency, late_finals, revisions };
            out.extend(recogniser(model, text, dur, rng, knobs, &timing));
            if vad {
                out.push(Verdict { at_ms: vad_end_at, word: Word::VadEnd });
            }
        } else if listening && speech.text.is_some() && model.empty_finals {
            out.extend(recogniser_noise(model, dur, rng, latency, &timing));
            if vad {
                out.push(Verdict { at_ms: vad_end_at, word: Word::VadEnd });
            }
        } else if vad {
            out.push(Verdict { at_ms: vad_end_at, word: Word::VadEnd });
        }
        // The realtime server hears the person with its own VAD, when it detects turns: its onset
        // and its end of speech come as late as measured, its end later still under
        // semantic detection, which waits for the words to sound finished.
        if let Some(semantic) = self.server_vad() {
            let (started, stopped) = if semantic {
                (SPEECH_STARTED_LAG_SEMANTIC, SPEECH_STOPPED_LAG_SEMANTIC)
            } else {
                (SPEECH_STARTED_LAG_SERVER, SPEECH_STOPPED_LAG_SERVER)
            };
            let started = delay(&mut self.rng, &started, &timing);
            let stopped = delay(&mut self.rng, &stopped, &timing);
            out.push(Verdict { at_ms: started, word: Word::RtStart });
            out.push(Verdict { at_ms: dur + stopped.max(1), word: Word::RtStop });
        }
        // Stable, so two verdicts due at the same instant keep the order chosen above.
        out.sort_by_key(|v| v.at_ms);
        out
    }

    /// Play one segment into the microphone beat by beat -- frames, a detector tick, the clock --
    /// saying each verdict once it falls due, and keep the frames flowing through the verdicts'
    /// tail: a real microphone never stops.
    ///
    /// Verdicts come after the beat's advance, because the driver only runs while the clock
    /// moves: the frames a verdict is about must have reached the providers before it is said.
    pub(super) async fn stretch(
        &mut self,
        mic: u32,
        seg: &Segment,
        verdicts: Vec<Verdict>,
    ) -> Result<(), Halt> {
        let stall_at = match seg.pacing {
            Pacing::Gap { .. } => self.rng.u64(0..seg.dur_ms.max(1)),
            _ => u64::MAX,
        };
        let mut verdicts = verdicts.into_iter().peekable();
        let mut elapsed = 0;
        let mut owed_ms = 0;
        let mut stalled = false;
        // Where the forwarding log stood when this segment began, and when its `SttStart` was
        // said. A provider knows only what the driver handed it: a recogniser does not start on,
        // or transcribe, audio it never received -- and the driver withholds the microphone from
        // it while an uninterruptible reply plays -- and a detector hears no speech in frames it
        // was not given. So every verdict waits until microphone audio has reached its stream,
        // and one still waiting when the speech is over is never said: that provider heard
        // silence, and silence has nothing to report.
        let segment_start = self.run.record.forwarded().len();
        let mut stt_start: Option<usize> = None;
        // An overlap: this speech starts while the agent's audio is coming out of the speaker.
        // Audio only handed to the output is not being spoken yet -- the driver marks the agent
        // speaking on the speaker's own started report -- so a segment merely live is not enough.
        // The detector is asked for its verdict one beat after `VadStart`, once the driver has
        // opened the overlap.
        let overlapping = seg.speaking && self.hearing();
        // The recogniser may give out partway through.
        let mut stt_fails_at = (self.rng.u8(0..100) < self.providers.stt_fails)
            .then(|| self.rng.u64(0..seg.dur_ms.max(1)));
        let mut vad_started_at: Option<u64> = None;
        let mut vad_ended = false;
        let mut judged = false;
        // Whose voice this stretch is: the person's, or now and then someone else in the room.
        self.current_speaker = if self.rng.u8(0..100) < self.persona.other_speaker {
            self.draw::<Option<Speaker>>()
        } else {
            self.speaker
        };
        while elapsed < seg.dur_ms || verdicts.peek().is_some() {
            let (advance, defer) = match seg.pacing {
                Pacing::Steady => (BEAT_MS, false),
                Pacing::Jitter { spread_ms } => {
                    (BEAT_MS - spread_ms + self.rng.u64(0..=2 * spread_ms), false)
                }
                Pacing::Gap { stall_ms } if elapsed >= stall_at && !stalled => {
                    stalled = true;
                    (stall_ms, true)
                }
                Pacing::Gap { .. } => (BEAT_MS, false),
            };
            // Land exactly on the segment's end, so the detector's verdict flips there.
            let advance =
                if elapsed < seg.dur_ms { advance.min(seg.dur_ms - elapsed) } else { advance };
            let advance = advance.max(1);
            owed_ms += advance;
            let chunks = if defer { 0 } else { owed_ms / FRAME_MS };
            owed_ms -= chunks * FRAME_MS;
            if chunks > 0 {
                let mic = Ref::new(mic);
                self.send(Input::MicFrames { mic, chunks: Count(chunks as u32) }).await?;
            }
            let was_speaking = seg.speaking && elapsed < seg.dur_ms;
            let beat_start = self.run.record.forwarded().len();
            self.advance(advance).await?;
            elapsed += advance;
            self.beat().await?;
            if stt_fails_at.is_some_and(|at| elapsed >= at)
                && let Some(stt) = self.first(Kind::Stt)
            {
                stt_fails_at = None;
                self.send(Input::SttFail { stt: Ref::new(stt) }).await?;
            }
            // The detector's tick on this beat's frames, once the driver has handed them over:
            // speech only in frames that carried the microphone, silence in anything else.
            if chunks > 0
                && let Some(vad) = self.first(Kind::Vad)
            {
                let speaking = was_speaking && self.reached(Kind::Vad, vad, beat_start);
                let p = self.rng.u8(if speaking { SPEAKING_P } else { SILENT_P });
                // What the detector reports: the speech's length while speaking, the silence's
                // since the speech ended otherwise.
                let duration = Millis(if speaking {
                    elapsed.min(seg.dur_ms)
                } else {
                    elapsed.saturating_sub(if seg.speaking { seg.dur_ms } else { 0 })
                });
                let vad = Ref::new(vad);
                self.send(Input::VadInference {
                    vad,
                    probability: Confidence(p),
                    speaking,
                    duration,
                })
                .await?;
            }
            // Due verdicts whose stream has heard the microphone are said; one still unheard once
            // the speech is over is dropped, along with everything that would have followed it.
            let over = elapsed >= seg.dur_ms;
            while let Some(verdict) = verdicts.next_if(|v| {
                v.at_ms <= elapsed && (over || self.sendable(&v.word, segment_start, stt_start))
            }) {
                if !self.sendable(&verdict.word, segment_start, stt_start) {
                    continue;
                }
                if matches!(verdict.word, Word::SttStart) {
                    stt_start = Some(self.run.record.forwarded().len());
                }
                if matches!(verdict.word, Word::VadStart) {
                    vad_started_at = Some(elapsed);
                }
                if matches!(verdict.word, Word::VadEnd) {
                    vad_ended = true;
                }
                self.say(seg, verdict).await?;
            }
            // A prediction the driver asked for shows up here too, after the advance.
            self.predict(seg.done).await?;
            // The detector answers when it has heard enough: a stretch long enough to be a
            // interruption is judged as soon as it is one, a shorter one once it has ended.
            let evidence = seg.dur_ms >= BACKCHANNEL_MS || vad_ended;
            if overlapping && !judged && evidence && vad_started_at.is_some_and(|at| elapsed > at) {
                let barge = seg.dur_ms >= BACKCHANNEL_MS;
                let right = self.rng.u8(0..100) < self.providers.detector_right;
                self.judge(barge == right).await?;
                judged = true;
            }
        }
        Ok(())
    }

    /// Send one verdict against what is live now; dropped when what it names is gone, because a
    /// recogniser whose utterance already closed has nobody to report to.
    async fn say(&mut self, seg: &Segment, verdict: Verdict) -> Result<(), Halt> {
        // The newest utterance: the one this segment's `SttStart` opened.
        let utterance = self.run.record.live().of(Kind::Utterance).into_iter().max().map(Ref::new);
        let vad = self.first(Kind::Vad).map(Ref::new);
        let stt = self.first(Kind::Stt).map(Ref::new);
        let input = match verdict.word {
            Word::VadStart => {
                vad.map(|vad| Input::VadStart { vad, speech_ms: Millis(verdict.at_ms) })
            }
            Word::VadEnd => vad.map(|vad| Input::VadEnd {
                vad,
                silence_ms: Millis(verdict.at_ms.saturating_sub(seg.dur_ms)),
            }),
            Word::SttStart => stt.map(|stt| Input::SttStart { stt }),
            Word::SttInterim(text) => utterance.map(|utterance| Input::SttInterim {
                utterance,
                text: UserText(text),
                speaker: self.current_speaker,
            }),
            Word::SttPreflight(text) => utterance.map(|utterance| Input::SttPreflight {
                utterance,
                text: UserText(text),
                speaker: self.current_speaker,
            }),
            Word::SttFinal(text) => utterance.map(|utterance| Input::SttFinal {
                utterance,
                text: UserText(text),
                speaker: self.current_speaker,
            }),
            Word::SttEnd => utterance.map(|utterance| Input::SttEnd { utterance }),
            Word::SttUsage => utterance.map(|utterance| Input::SttUsage { utterance }),
            Word::RtStart => self.first(Kind::Realtime).map(|n| Input::RealtimeEvent {
                socket: Ref::new(n),
                event: RtEvent::SpeechStarted,
            }),
            Word::RtStop => self.first(Kind::Realtime).map(|n| Input::RealtimeEvent {
                socket: Ref::new(n),
                event: RtEvent::SpeechStopped,
            }),
        };
        match input {
            Some(input) => self.send(input).await,
            None => Ok(()),
        }
    }

    /// Whether a verdict's stream has heard what the verdict is about: the detector's, the
    /// microphone since the segment began; the recogniser's start, the same; the recogniser's
    /// words, the microphone since the start that opened the utterance.
    fn sendable(&self, word: &Word, segment_start: usize, stt_start: Option<usize>) -> bool {
        match word {
            Word::VadStart | Word::VadEnd => {
                self.first(Kind::Vad).is_some_and(|vad| self.reached(Kind::Vad, vad, segment_start))
            }
            Word::SttStart => {
                self.first(Kind::Stt).is_some_and(|stt| self.reached(Kind::Stt, stt, segment_start))
            }
            Word::RtStart | Word::RtStop => self
                .first(Kind::Realtime)
                .is_some_and(|rt| self.reached(Kind::Realtime, rt, segment_start)),
            // A recogniser reports the words it heard: audio since the segment began, and an
            // utterance open to report them on (its start comes with its first words).
            Word::SttInterim(_)
            | Word::SttPreflight(_)
            | Word::SttFinal(_)
            | Word::SttEnd
            | Word::SttUsage => {
                stt_start.is_some()
                    && self
                        .first(Kind::Stt)
                        .is_some_and(|stt| self.reached(Kind::Stt, stt, segment_start))
            }
        }
    }

    /// Whether microphone audio has reached this provider stream since a point in the forwarding
    /// log. The driver's own record of what it handed over, not the case's record of what it sent.
    fn reached(&self, kind: Kind, nth: u32, since: usize) -> bool {
        self.run.record.forwarded()[since..].iter().any(|(_, k, n, tags)| {
            *k == kind && *n == nth && tags.iter().any(|tag| matches!(tag, Tag::Mic { .. }))
        })
    }
}

/// The listener's side: judging an overlap, answering the predictor (hearing the agent is the
/// speaker's, in `speaker`).
impl Talk {
    /// The interruption detector's verdict on the overlap in progress, if there is a detector.
    async fn judge(&mut self, is_interruption: bool) -> Result<(), Halt> {
        let Some(port) = self.first(Kind::Interruption) else { return Ok(()) };
        let port = Ref::new(port);
        if self.rng.u8(0..100) < self.providers.overlap_fails {
            return self.send(Input::OverlapFail { port }).await;
        }
        // Whether the agent had already stopped by the time the detector answered is the
        // detector's to report, and either answer is legal against any driver state.
        let agent_ended = self.rng.bool();
        self.send(Input::OverlapVerdict { port, is_interruption, agent_ended }).await
    }

    /// Answer the end-of-turn predictor's newest pending request, if the driver made one:
    /// confident when the person is done, doubtful when they are only pausing. That is the
    /// predictor's job, and the driver's endpointing is what its answer feeds.
    pub(super) async fn predict(&mut self, done: bool) -> Result<(), Halt> {
        let pending = self.run.record.live().of(Kind::Prediction).into_iter().max();
        let Some(nth) = pending.filter(|nth| !self.unanswered.contains(nth)) else {
            return Ok(());
        };
        let prediction = Ref::new(nth);
        if self.rng.u8(0..100) < self.providers.eot_lost {
            if self.rng.bool() {
                self.unanswered.insert(nth);
                return Ok(());
            }
            return self.send(Input::EotFail { prediction }).await;
        }
        let probability = Confidence(self.rng.u8(if done { EOT_DONE_P } else { EOT_MORE_P }));
        let backchannel = self.draw::<Option<Confidence>>();
        self.send(Input::EotAnswer { prediction, probability, backchannel }).await
    }
}
