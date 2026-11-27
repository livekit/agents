use super::*;

use crate::calibration::llm::{LLM_CHARS_PER_CHUNK, LLM_CHUNK_GAP, LLM_FIRST_CHUNK};

pub use crate::calibration::stt::{STT_MODELS, SttModel};
use crate::fakes::SttSpeech;

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

    /// What the model sends on one chat request, in order. A request offering the input router's or
    /// keyterm detector's tool is answered with it;
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
        let input_candidates = match self.run.record.resource(Kind::Chat, chat).map(|r| r.detail())
        {
            Some(Detail::Chat { input_candidates, .. }) => *input_candidates,
            _ => None,
        };
        let tool_call = if let Some(candidates) = input_candidates {
            // Untrusted provider choices: no-match, one/several matches, unclear, or an
            // out-of-range reference. The actual parser and owner decide their admission.
            Some((
                lk_agents_core::session::input_routing::TOOL.into(),
                input_verdict(candidates, self.rng.usize(0..5)),
            ))
        } else if offered(&self.run, lk_agents_core::delegation::remote::routing::TOOL) {
            // The remote call's router: new work, a continuation or a held response of one of
            // the first candidates, unclear, or indices the owner must refuse.
            let verdict = REMOTE_VERDICTS[self.rng.usize(0..REMOTE_VERDICTS.len())];
            Some((
                lk_agents_core::delegation::remote::routing::TOOL.into(),
                ToolArgs(verdict.into()),
            ))
        } else if offered(&self.run, RECORD_KEYTERMS_TOOL) {
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
        } else if !self.run.setup.tools.is_empty() && self.rng.u8(0..100) < self.host.tool_calls {
            let tools = &self.run.setup.tools;
            let name = tools[self.rng.usize(0..tools.len())].clone();
            if name == crate::fakes::remote_tool::REMOTE_TOOL {
                // A request stated in full, or arguments the delegation refuses.
                let args = REMOTE_REQUESTS[self.rng.usize(0..REMOTE_REQUESTS.len())];
                Some((name, ToolArgs(args.into())))
            } else {
                Some((name, self.draw::<ToolArgs>()))
            }
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

/// Provider payload fixtures, selected independently of the person's words. A routing fake
/// never decides which request owns an answer or whether an operation is approved.
/// What the frontend asks the remote desk for, as a model states it.
const REMOTE_REQUESTS: [&str; 4] = [
    r#"{"request":"Book a table for two tonight"}"#,
    r#"{"request":"Move my booking to Friday"}"#,
    r#"{"request":"Is my booking confirmed?"}"#,
    "{}",
];

/// What the remote call router answers.
const REMOTE_VERDICTS: [&str; 7] = [
    r#"{"action":"new","candidate":null,"response":null}"#,
    r#"{"action":"continue","candidate":0,"response":null}"#,
    r#"{"action":"respond","candidate":0,"response":0}"#,
    r#"{"action":"respond","candidate":0,"response":0}"#,
    r#"{"action":"respond","candidate":1,"response":0}"#,
    r#"{"action":"unclear","candidate":null,"response":null}"#,
    r#"{"action":"respond","candidate":7,"response":null}"#,
];

fn input_verdict(candidates: usize, choice: usize) -> ToolArgs {
    let (selected, unclear) = match choice {
        0 => (Vec::new(), false),
        1 => (vec![0], false),
        2 => ((0..candidates).collect(), false),
        3 => (Vec::new(), true),
        _ => (vec![candidates], false),
    };
    ToolArgs(serde_json::json!({"candidates":selected, "unclear":unclear}).to_string())
}

#[cfg(test)]
#[path = "providers_tests.rs"]
mod tests;

/// The person's provider knobs a recogniser's verdicts depend on.
#[derive(Clone, Copy)]
pub(super) struct SttKnobs {
    /// How the recogniser reports speech: its own start and end, and when it transcribes.
    pub(super) reports: SttSpeech,
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
///
/// A recogniser that detects speech itself reports its end either after the final or, as its own
/// detector hears the person stop, before it: always so for one that transcribes a speech only
/// once it ended, and otherwise only once it has words in hand (a provider that transcribes
/// during the speech has sent some by its end). Without its own detection, its start and end are
/// the case's bookkeeping only: the stream's speech starts with the words and ends with a final.
pub(super) fn recogniser(
    model: &SttModel,
    text: &str,
    dur: u64,
    rng: &mut Rng,
    knobs: SttKnobs,
    timing: &Timing,
) -> Vec<Verdict> {
    let SttKnobs { reports, latency, late_finals, revisions } = knobs;
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
        let worded = first_at < final_at;
        let end_first = reports.provider_vad
            && (reports.transcribes_after_speech_ends || (worded && rng.u8(0..100) < 40));
        if end_first {
            // Its detector's end, once the person stopped: after its start (and the first words
            // it holds the end for), never after the final.
            let heard_end = spoken_to + slow(rng.u64(VAD_END_MS));
            let at_ms = heard_end.clamp(first_at.min(final_at), final_at);
            out.push(Verdict { at_ms, word: Word::SttEnd });
        }
        out.push(Verdict { at_ms: final_at, word: Word::SttUsage });
        out.push(Verdict { at_ms: final_at, word: Word::SttFinal(piece.clone()) });
        if !end_first {
            out.push(Verdict { at_ms: final_at, word: Word::SttEnd });
        }
        last_end = final_at;
        // AssemblyAI's formatted turn: the same words again, a moment later, as an
        // utterance of their own.
        if rng.u64(0..of) < dup {
            let again = final_at + delay(rng, &model.duplicate_final_after, timing).max(1);
            // One that detects speech may send it with no start of its own: words outside its
            // speech, its end after them.
            if !reports.provider_vad || rng.bool() {
                out.push(Verdict { at_ms: again, word: Word::SttStart });
            }
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
    knobs: SttKnobs,
    timing: &Timing,
) -> Vec<Verdict> {
    if !model.empty_finals {
        return Vec::new();
    }
    let at_ms = dur + delay(rng, &model.final_after_end, timing) * knobs.latency / 100;
    let usage = [Word::SttUsage, Word::SttFinal(String::new())];
    // Its detector may close the noise before the empty final does.
    let words: Vec<Word> = if knobs.reports.provider_vad && rng.bool() {
        [Word::SttStart, Word::SttEnd].into_iter().chain(usage).collect()
    } else {
        [Word::SttStart].into_iter().chain(usage).chain([Word::SttEnd]).collect()
    };
    words.into_iter().map(|word| Verdict { at_ms, word }).collect()
}
