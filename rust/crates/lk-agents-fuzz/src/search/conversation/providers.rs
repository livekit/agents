//! Model streaming and recorded speech recognition behavior.

use super::*;

// A real model's stream as measured (`calibration.json`, `llm/openai/gpt-4.1-mini/<name>`): its
// first chunk's delay and the gap between chunks in ms, the characters in a chunk.
const LLM_FIRST_CHUNK: Measured = [372, 445, 479, 519, 547, 597, 694, 746, 769, 999, 3117];
const LLM_CHUNK_GAP: Measured = [0, 0, 0, 2, 4, 6, 8, 11, 18, 27, 172];
const LLM_CHARS_PER_CHUNK: Measured = [1, 2, 3, 4, 4, 5, 6, 7, 8, 8, 18];

/// A real recogniser as measured (`calibration.json`, `stt/<name>/<field>`): its delays in ms, how
/// it cuts a turn, and its quirks. A quantity a model never showed is all zeros and never drawn.
pub struct SttModel {
    pub name: &'static str,
    first_after_onset: Measured,
    interim_gap: Measured,
    final_after_end: Measured,
    finals_per_utterance: Measured,
    preflights_per_utterance: Measured,
    final_after_preflight: Measured,
    /// Utterances it finalised twice, of all measured.
    duplicate_share: (u64, u64),
    duplicate_final_after: Measured,
    /// It opens an utterance over noise and closes it with an empty final.
    pub empty_finals: bool,
}

/// The recognisers a setup's `stt_model` picks from.
pub const STT_MODELS: [SttModel; 3] = [
    SttModel {
        name: "deepgram/nova-3",
        first_after_onset: [660, 761, 774, 782, 817, 829, 836, 1016, 1391, 1738, 1903],
        interim_gap: [802, 950, 968, 970, 972, 978, 984, 996, 1020, 1020, 1023],
        final_after_end: [57, 245, 270, 281, 306, 325, 332, 344, 353, 365, 546],
        finals_per_utterance: [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 4],
        preflights_per_utterance: [0; 11],
        final_after_preflight: [0; 11],
        duplicate_share: (3, 28),
        duplicate_final_after: [82, 101, 120, 140, 159, 178, 185, 193, 200, 208, 215],
        empty_finals: true,
    },
    SttModel {
        name: "assemblyai/universal-streaming",
        first_after_onset: [765, 836, 880, 898, 922, 1023, 1032, 1133, 1186, 1572, 1823],
        interim_gap: [40, 46, 92, 103, 138, 180, 218, 269, 349, 554, 967],
        final_after_end: [357, 370, 376, 499, 826, 930, 945, 981, 1031, 1046, 1073],
        finals_per_utterance: [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2],
        preflights_per_utterance: [0; 11],
        final_after_preflight: [0; 11],
        duplicate_share: (5, 22),
        duplicate_final_after: [138, 139, 140, 149, 168, 187, 200, 213, 221, 225, 229],
        empty_finals: false,
    },
    SttModel {
        name: "deepgram/flux-general-en",
        first_after_onset: [425, 506, 588, 607, 626, 736, 845, 866, 886, 977, 1068],
        interim_gap: [747, 751, 755, 758, 762, 766, 799, 832, 864, 897, 930],
        final_after_end: [154, 184, 213, 239, 265, 386, 507, 734, 960, 983, 1006],
        finals_per_utterance: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
        preflights_per_utterance: [1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4],
        final_after_preflight: [0, 0, 2, 6, 12, 19, 56, 121, 240, 312, 359],
        duplicate_share: (0, 7),
        duplicate_final_after: [0; 11],
        empty_finals: false,
    },
];

/// The model's side: every chat request is answered on the model's own clock, as a real one
/// answers -- the first chunk after a while, then token-sized pieces a few milliseconds apart,
/// then the end -- while the person and the developer go on between the chunks.
impl Talk {
    /// Plan the answer to every chat request opened since the last look: what the model says and
    /// when each piece of it arrives.
    pub(super) fn plan_chats(&mut self) {
        let opened: Vec<u32> = self
            .run
            .record
            .live()
            .of(Kind::Chat)
            .into_iter()
            .filter(|chat| !self.planned.contains(chat))
            .collect();
        let latency = self.providers.latency;
        let timing = self.run.setup.timing();
        for chat in opened {
            self.planned.insert(chat);
            let deliveries = self.answer(chat);
            let first = delay(&mut self.rng, &LLM_FIRST_CHUNK, &timing);
            let mut at = self.now_ms() + first * latency / 100;
            let mut due = VecDeque::new();
            for (n, input) in deliveries.into_iter().enumerate() {
                if n > 0 {
                    at += delay(&mut self.rng, &LLM_CHUNK_GAP, &timing) * latency / 100;
                }
                due.push_back((at, input));
            }
            self.streams.insert(chat, due);
        }
    }

    /// What the model sends on one chat request, in order. A request offering the
    /// answering-machine classifier's or the keyterm detector's tool is answered with that tool;
    /// otherwise the model reaches for one of the agent's tools or says one to three of the
    /// agent's lines. A failing model fails anywhere in it, before its first chunk or partway.
    fn answer(&mut self, chat: u32) -> Vec<Input> {
        let at = Ref::new(chat);
        let offered = |run: &Run, tool: &str| {
            matches!(
                run.record.resource(Kind::Chat, chat).map(|r| r.detail()),
                Some(Detail::Chat { tools, .. }) if tools.iter().any(|t| t == tool)
            )
        };
        let tool_call = if offered(&self.run, RECORD_KEYTERMS_TOOL) {
            // Terms to track, to apply and to drop, drawn from the call's names, or an answer
            // that changes nothing.
            let mut terms = || -> Vec<&str> {
                let n = self.rng.usize(0..3);
                (0..n).map(|_| NAMES[self.rng.usize(0..NAMES.len())]).collect()
            };
            let (pending, confirm, remove) = (terms(), terms(), terms());
            let args = if self.rng.u8(0..100) < 15 {
                "{}".to_owned()
            } else {
                serde_json::json!({ "pending": pending, "confirm": confirm, "remove": remove })
                    .to_string()
            };
            Some((RECORD_KEYTERMS_TOOL.to_owned(), ToolArgs(args)))
        } else if offered(&self.run, SAVE_PREDICTION_TOOL) {
            // Each label the tool accepts, a postponement (one past the clamp included), and an
            // answer without the label, which the classifier must ignore.
            let choices = AmdCategory::ALL.len() + 2;
            let (name, args) = match self.rng.usize(0..choices) {
                i if i < AmdCategory::ALL.len() => (
                    SAVE_PREDICTION_TOOL,
                    format!(r#"{{"label":"{}"}}"#, AmdCategory::ALL[i].as_str()),
                ),
                i if i == AmdCategory::ALL.len() => (
                    POSTPONE_TERMINATION_TOOL,
                    format!(r#"{{"seconds":{}}}"#, self.rng.u64(0..=MAX_EXTENSION_SECS as u64 + 2)),
                ),
                _ => (SAVE_PREDICTION_TOOL, "{}".to_owned()),
            };
            Some((name.to_owned(), ToolArgs(args)))
        } else if !self.run.setup.tools.is_empty() && self.rng.u8(0..100) < self.host.tool_calls {
            let tools = &self.run.setup.tools;
            let name = tools[self.rng.usize(0..tools.len())].clone();
            Some((name, self.draw::<ToolArgs>()))
        } else {
            None
        };
        let mut out = match tool_call {
            // A tool call comes whole, in one chunk.
            Some((name, args)) => vec![Input::LlmToolCall { chat: at, name, args }],
            None => {
                let lines: Vec<String> =
                    (0..self.rng.usize(1..=3)).map(|_| self.line().0).collect();
                let text = lines.join(" ");
                self.turn_words += text.split_whitespace().count() as u64;
                self.pieces(&text)
                    .into_iter()
                    .map(|piece| Input::LlmText { chat: at, text: AgentText(piece) })
                    .collect()
            }
        };
        out.push(Input::LlmEnd { chat: at });
        if self.rng.u8(0..100) < self.providers.llm_fails {
            out.truncate(self.rng.usize(0..out.len()));
            out.push(Input::LlmFail { chat: at });
        }
        out
    }

    /// A reply cut into the pieces the model streams it in: a real model's characters per chunk.
    fn pieces(&mut self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        if chars.is_empty() {
            return vec![String::new()];
        }
        let mut out = Vec::new();
        let mut at = 0;
        while at < chars.len() {
            let n = (measured(&mut self.rng, &LLM_CHARS_PER_CHUNK) as usize).max(1);
            let end = (at + n).min(chars.len());
            out.push(chars[at..end].iter().collect());
            at = end;
        }
        out
    }

    /// Let `ms` of virtual time pass, stopping at every instant a model chunk is due to deliver
    /// it: the driver sees each piece when the model sends it, not in a lump at the next beat.
    pub(super) async fn advance(&mut self, ms: u64) -> Result<(), Halt> {
        let end = self.now_ms() + ms;
        loop {
            self.plan_chats();
            let now = self.now_ms();
            let next = self.streams.values().filter_map(|due| due.front().map(|(at, _)| *at)).min();
            let Some(at) = next.filter(|at| *at <= end) else {
                if end > now {
                    self.send(Input::Advance { ms: Millis(end - now) }).await?;
                }
                return Ok(());
            };
            if at > now {
                self.send(Input::Advance { ms: Millis(at - now) }).await?;
            }
            self.deliver_due().await?;
        }
    }

    /// Send every model delivery due by now. A request the driver dropped -- cancelled by an
    /// interruption, its speech gone -- has nobody to stream to, and the rest of its answer goes.
    async fn deliver_due(&mut self) -> Result<(), Halt> {
        let now = self.now_ms();
        let live = self.run.record.live().of(Kind::Chat);
        self.streams.retain(|chat, _| live.contains(chat));
        let mut due = Vec::new();
        for queue in self.streams.values_mut() {
            while queue.front().is_some_and(|(at, _)| *at <= now) {
                due.extend(queue.pop_front().map(|(_, input)| input));
            }
        }
        self.streams.retain(|_, queue| !queue.is_empty());
        for input in due {
            self.delivered += 1;
            self.send(input).await?;
        }
        Ok(())
    }
}

/// The person's provider knobs a recogniser's verdicts depend on.
#[derive(Clone, Copy)]
pub(super) struct SttKnobs {
    /// In percent of the recorded delays.
    pub(super) latency: u64,
    /// How often the last final comes 1.5-3 s late (a provider under load, past what was
    /// measured).
    pub(super) late_finals: u8,
    /// How often an interim is mis-heard, then corrected.
    pub(super) revisions: u8,
}

/// What a real recogniser says about `text` spoken over `dur` ms, as `model` was measured doing:
/// when its first words come, how often it restates them, how it cuts a long turn into finals,
/// what it sends around each final (a usage report before it, the end of speech with it), and
/// its quirks: AssemblyAI's second final for the same words, Flux's preflights. Measured from the
/// speech's start.
pub(super) fn recogniser(
    model: &SttModel,
    text: &str,
    dur: u64,
    rng: &mut Rng,
    knobs: SttKnobs,
    timing: &Timing,
) -> Vec<Verdict> {
    let SttKnobs { latency, late_finals, revisions } = knobs;
    let slow = |ms: u64| ms * latency / 100;
    let mut out = Vec::new();
    let words: Vec<&str> = text.split(' ').collect();
    let n = words.len().max(1);
    let pieces = (measured(rng, &model.finals_per_utterance) as usize).clamp(1, n);
    let preflights = measured(rng, &model.preflights_per_utterance) as usize;
    let (dup, of) = model.duplicate_share;
    let revised_piece = (rng.u8(0..100) < revisions).then(|| rng.usize(0..pieces));
    let mut last_end = 0;
    for p in 0..pieces {
        let (a, b) = (n * p / pieces, n * (p + 1) / pieces);
        let piece = words[a.min(words.len())..b.min(words.len())].join(" ");
        let spoken_from = dur * a as u64 / n as u64;
        let spoken_to = if p + 1 == pieces { dur } else { dur * b as u64 / n as u64 };
        // A provider under load answers late now and then (`late_finals`, past what was
        // measured); usually its tail is the measured one.
        let tail = if p + 1 == pieces && rng.u8(0..100) < late_finals {
            rng.u64(STT_LATE_FINAL_MS)
        } else {
            slow(delay(rng, &model.final_after_end, timing))
        };
        let first_at =
            (spoken_from + slow(delay(rng, &model.first_after_onset, timing))).max(last_end + 1);
        // No words before the recogniser has any: a short piece's final waits for its first.
        let final_at = (spoken_to + tail).max(last_end + 1).max(first_at);
        // Its start of speech comes with its first words, never before.
        if first_at < final_at {
            out.push(Verdict { at_ms: first_at, word: Word::SttStart });
            // The words heard by `t`, as a provider restates them.
            let heard_by = |t: u64| {
                let heard = (t.saturating_sub(spoken_from) * (b - a) as u64
                    / spoken_to.saturating_sub(spoken_from).max(1))
                .clamp(1, (b - a).max(1) as u64) as usize;
                words[a..(a + heard).min(words.len())].to_vec()
            };
            // Flux's preflights: the words so far, judged finished, the last just before the
            // final and nothing restated after it.
            let last_preflight = (preflights > 0).then(|| {
                final_at
                    .saturating_sub(delay(rng, &model.final_after_preflight, timing))
                    .max(first_at)
            });
            let mut t = first_at;
            let mut nth = 0;
            while t < last_preflight.unwrap_or(final_at) || nth == 0 {
                let mut prefix = heard_by(t);
                if revised_piece == Some(p) && nth == 1 && !prefix.is_empty() {
                    // The last word so far, mis-heard as a word from some other turn.
                    let other = USER_TURNS[rng.usize(0..USER_TURNS.len())];
                    let stray = other
                        .split(' ')
                        .nth(rng.usize(0..other.split(' ').count()))
                        .unwrap_or("uh");
                    *prefix.last_mut().unwrap() = stray;
                }
                out.push(Verdict { at_ms: t, word: Word::SttInterim(prefix.join(" ")) });
                nth += 1;
                t += delay(rng, &model.interim_gap, timing).max(1);
            }
            if let Some(last) = last_preflight {
                // Every final has its preflight; the last piece the turn's whole run of them.
                let count = if p + 1 == pieces { preflights } else { 1 };
                for k in 0..count {
                    let at_ms = first_at + (last - first_at) * (k as u64 + 1) / count as u64;
                    let said =
                        if k + 1 == count { piece.clone() } else { heard_by(at_ms).join(" ") };
                    out.push(Verdict { at_ms, word: Word::SttPreflight(said) });
                }
            }
        } else {
            out.push(Verdict { at_ms: final_at, word: Word::SttStart });
            // A model that judges every turn's end restates the words and its verdict first,
            // however short the turn.
            if preflights > 0 {
                out.push(Verdict { at_ms: final_at, word: Word::SttInterim(piece.clone()) });
                out.push(Verdict { at_ms: final_at, word: Word::SttPreflight(piece.clone()) });
            }
        }
        out.push(Verdict { at_ms: final_at, word: Word::SttUsage });
        out.push(Verdict { at_ms: final_at, word: Word::SttFinal(piece.clone()) });
        out.push(Verdict { at_ms: final_at, word: Word::SttEnd });
        last_end = final_at;
        // AssemblyAI's formatted turn: the same words again, a moment later, as an
        // utterance of their own.
        if rng.u64(0..of) < dup {
            let again = final_at + delay(rng, &model.duplicate_final_after, timing).max(1);
            out.push(Verdict { at_ms: again, word: Word::SttStart });
            out.push(Verdict { at_ms: again, word: Word::SttUsage });
            out.push(Verdict { at_ms: again, word: Word::SttFinal(piece) });
            out.push(Verdict { at_ms: again, word: Word::SttEnd });
            last_end = again;
        }
    }
    out
}

/// Noise to a recogniser that reports it: Deepgram opens an utterance over it and closes it with
/// an empty final.
pub(super) fn recogniser_noise(
    model: &SttModel,
    dur: u64,
    rng: &mut Rng,
    latency: u64,
    timing: &Timing,
) -> Vec<Verdict> {
    if !model.empty_finals {
        return Vec::new();
    }
    let at_ms = dur + delay(rng, &model.final_after_end, timing) * latency / 100;
    [Word::SttStart, Word::SttUsage, Word::SttFinal(String::new()), Word::SttEnd]
        .into_iter()
        .map(|word| Verdict { at_ms, word })
        .collect()
}
