//! Transcript sync accuracy against recorded speech. A recording is a real synthesis: the reply's
//! deltas, each audio frame at the time it arrived, its audio and the voice's own
//! word timings. The replay feeds the pacer as a session does (the text ahead of the audio, each
//! frame as it arrives, the pauses the detector finds in it, the playhead moving in real time)
//! and scores when each word was shown against when the voice said it.
//!
//! `transcript_sync_holds_its_accuracy` replays the replies checked in beside the module
//! (`sync_recordings/`, Opus, so it needs `--features ffmpeg`); `make bench-sync` measures the
//! full recorded corpus and tracks its drift across commits.

use super::*;
use crate::transcription::{SegmentText, effects::TranscriptSyncEffect};
use lk_agents_abi::tts::markup::Provider;
use lk_agents_core::lexer::Vocabulary;
use lk_agents_media::pauses::PauseDetector;

/// How often the playhead is read, in seconds: the synchroniser ticks at each word's time, so
/// this only bounds the replay's own rounding.
const TICK: f64 = 0.005;

/// One recorded reply.
struct Recording {
    model: String,
    deltas: Vec<String>,
    /// Each frame's arrival, seconds since the stream opened, and its duration.
    frames: Vec<(f64, f64)>,
    sample_rate: u32,
    pcm: Vec<i16>,
    /// The voice's words, as the transcript's tokens, each with when the voice started it.
    truth: Vec<(String, f64)>,
}

fn vocab(model: &str) -> Vocabulary {
    let provider = if model.starts_with("cartesia/") {
        Some(Provider::Cartesia)
    } else if model.starts_with("inworld/") && model.contains("tts-2") {
        Some(Provider::Inworld)
    } else {
        None
    };
    Vocabulary::of(provider)
}

/// Each spoken piece the pacer forwarded, with the playhead when it was shown.
fn replay(r: &Recording) -> Vec<(String, f64)> {
    let mut text = SegmentText::new(vocab(&r.model));
    let mut chunks: Vec<TranscriptChunk> = Vec::new();
    for delta in &r.deltas {
        chunks.extend(text.push(delta).shown);
    }
    chunks.extend(text.finish().shown);

    let mut out = Vec::new();
    let mut position = 0.0;
    let step = |pacer: TranscriptPacer, ev, at: f64, out: &mut Vec<(String, f64)>| {
        let (next, effects) = pacer.step(ev);
        for effect in effects {
            if let TranscriptSyncEffect::PublishWord { chunk, .. } = effect
                && chunk.spoken
            {
                out.push((chunk.text, at));
            }
        }
        next
    };
    let mut pacer = TranscriptPacer::new(TranscriptPacerOptions::default());
    // the reply's text is ahead of its audio, as a model's is of a voice's
    for chunk in chunks {
        let ev = TranscriptPacerEvent::Text { chunk, start_time: None, end_time: None };
        pacer = step(pacer, ev, position, &mut out);
    }
    pacer = step(pacer, TranscriptPacerEvent::TextEnd, position, &mut out);

    // the sink takes each frame as it arrives; the playout starts with the first and plays on,
    // never past the audio received
    let first = r.frames.first().map_or(0.0, |f| f.0);
    let total: f64 = r.frames.iter().map(|f| f.1).sum();
    let mut detector = PauseDetector::new(r.sample_rate, 1);
    let (mut received, mut next_frame, mut fed, mut t) = (0.0, 0, 0, 0.0);
    while t <= total + TICK {
        while next_frame < r.frames.len() && r.frames[next_frame].0 - first <= t {
            let secs = r.frames[next_frame].1;
            pacer = step(pacer, TranscriptPacerEvent::Audio { secs }, position, &mut out);
            received += secs;
            next_frame += 1;
            // the detector reads this frame's samples
            let to = ((received * f64::from(r.sample_rate)).round() as usize).min(r.pcm.len());
            for p in detector.push(r.pcm[fed.min(to)..to].iter().copied()) {
                let ev = TranscriptPacerEvent::Pause { start: p.start, end: p.end };
                pacer = step(pacer, ev, position, &mut out);
            }
            fed = to;
            if next_frame == r.frames.len() {
                if let Some(p) = detector.finish() {
                    let ev = TranscriptPacerEvent::Pause { start: p.start, end: p.end };
                    pacer = step(pacer, ev, position, &mut out);
                }
                pacer = step(pacer, TranscriptPacerEvent::AudioEnd, position, &mut out);
            }
        }
        position = f64::min(t, received);
        let ev = TranscriptPacerEvent::Position(Duration::from_secs_f64(position));
        pacer = step(pacer, ev, position, &mut out);
        t += TICK;
    }
    let ev = TranscriptPacerEvent::PlaybackFinished {
        position: Duration::from_secs_f64(total),
        interrupted: false,
    };
    step(pacer, ev, total, &mut out);
    out
}

/// A word as both sides are compared: letters and digits, lowercase.
fn norm(text: &str) -> String {
    text.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

fn is_cjk(c: char) -> bool {
    ('\u{3040}'..='\u{30ff}').contains(&c) || ('\u{3400}'..='\u{9fff}').contains(&c)
}

/// The shown pieces as tokens (a CJK character each), a word split across pieces joined, each
/// with when it was shown.
fn shown_tokens(pieces: &[(String, f64)]) -> Vec<(String, f64)> {
    let mut out: Vec<(String, f64)> = Vec::new();
    let mut glued = false;
    for (text, at) in pieces {
        for (i, word) in text.split_whitespace().enumerate() {
            let cjk = word.chars().any(is_cjk);
            let joins = i == 0 && glued && !text.starts_with(char::is_whitespace) && !cjk;
            if cjk {
                let chars = word.chars().filter(|c| c.is_alphanumeric());
                out.extend(chars.map(|c| (norm(&c.to_string()), *at)));
            } else if joins && let Some(last) = out.last_mut() {
                last.0.push_str(&norm(word));
            } else if !norm(word).is_empty() {
                out.push((norm(word), *at));
            }
        }
        glued = !text.ends_with(char::is_whitespace) && !text.is_empty();
    }
    out
}

/// |shown - said| for the words both sides have, matched in order (longest common subsequence).
fn errors(shown: &[(String, f64)], truth: &[(String, f64)]) -> Vec<f64> {
    let (n, m) = (shown.len(), truth.len());
    let mut lcs = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if shown[i].0 == truth[j].0 {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }
    let (mut i, mut j, mut out) = (0, 0, Vec::new());
    while i < n && j < m {
        if shown[i].0 == truth[j].0 {
            out.push((shown[i].1 - truth[j].1).abs());
            (i, j) = (i + 1, j + 1);
        } else if lcs[i + 1][j] >= lcs[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

fn percentiles(mut e: Vec<f64>) -> (f64, f64, f64, usize) {
    e.sort_by(f64::total_cmp);
    let at = |q: f64| e[((e.len() as f64 * q) as usize).min(e.len() - 1)];
    (at(0.5), at(0.9), at(0.99), e.len())
}

/// A reply as recorded: its audio beside it, the voice's timings as stamps.
#[derive(serde::Deserialize)]
struct Sample {
    model: String,
    deltas: Vec<String>,
    frames: Vec<(f64, f64)>,
    sample_rate: u32,
    stamps: Vec<Stamp>,
}

#[derive(serde::Deserialize)]
struct Stamp {
    text: String,
    start: f64,
    end: f64,
}

/// The voice's words as tokens, a stamp of several spread over its span.
fn truth(stamps: &[Stamp]) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    for s in stamps {
        let tokens: Vec<String> = s
            .text
            .split_whitespace()
            .flat_map(|w| {
                if w.chars().any(is_cjk) {
                    let chars = w.chars().filter(|c| c.is_alphanumeric());
                    chars.map(|c| norm(&c.to_string())).collect()
                } else {
                    vec![norm(w)]
                }
            })
            .filter(|t| !t.is_empty())
            .collect();
        let span = (s.end - s.start) / tokens.len().max(1) as f64;
        out.extend(tokens.into_iter().enumerate().map(|(k, t)| (t, s.start + k as f64 * span)));
    }
    out
}

fn recording(sample: Sample, pcm: Vec<i16>) -> Recording {
    Recording {
        pcm,
        truth: truth(&sample.stamps),
        sample_rate: sample.sample_rate,
        model: sample.model,
        deltas: sample.deltas,
        frames: sample.frames,
    }
}

/// One corpus reply: a directory of `sample.json` and its 16-bit mono `audio.pcm`.
#[allow(clippy::disallowed_methods, reason = "a measurement over recorded files")]
fn load(dir: &std::path::Path) -> Option<Recording> {
    let sample: Sample =
        serde_json::from_slice(&std::fs::read(dir.join("sample.json")).ok()?).ok()?;
    let raw = std::fs::read(dir.join("audio.pcm")).ok()?;
    Some(recording(sample, raw.as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes(*b)).collect()))
}

/// The checked-in replies: Cartesia, Inworld and ElevenLabs voices; plain, numbers, markdown,
/// heavy punctuation and expressive replies in English, and a reply each in French, Spanish,
/// German, Hindi, Chinese and Japanese. Each is its `.json` beside its `.opus`.
#[cfg(feature = "ffmpeg")]
#[allow(clippy::disallowed_methods, reason = "test fixtures read from disk")]
fn checked_in() -> Vec<Recording> {
    use lk_agents_media::{Container, audio_decoder::DecodeLimits, ffmpeg::decode_stream};
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/transcription/sync_recordings");
    let mut jsons: Vec<_> = std::fs::read_dir(&dir)
        .expect("sync_recordings")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    jsons.sort();
    let limits = DecodeLimits {
        min_sample_rate: 8000,
        max_sample_rate: 48000,
        max_channels: 2,
        max_streams: 1,
        max_expansion_ratio: 64,
        max_samples_per_frame: 65536,
    };
    jsons
        .iter()
        .map(|json| {
            let sample: Sample =
                serde_json::from_slice(&std::fs::read(json).expect("json")).expect("sample");
            let opus = std::fs::read(json.with_extension("opus")).expect("opus");
            let mut pcm = Vec::new();
            let rate = sample.sample_rate;
            decode_stream(Some(Container::Ogg), rate, 1, limits, &mut opus.as_slice(), |_, s| {
                pcm.extend_from_slice(s);
                Ok(())
            })
            .expect("the opus decodes");
            recording(sample, pcm)
        })
        .collect()
}

// Catches: a change to the pause detector, the estimator or the pacer's wiring (a break lost, a
// word mis-tokenized) that shows words further from when the voice says them. The limits are the
// checked-in replies' measured accuracy with some room; `make bench-sync` is the finer measure.
#[cfg(feature = "ffmpeg")]
#[test]
fn transcript_sync_holds_its_accuracy() {
    let recordings = checked_in();
    let e: Vec<f64> =
        recordings.iter().flat_map(|r| errors(&shown_tokens(&replay(r)), &r.truth)).collect();
    let (median, p90, p99, n) = percentiles(e);
    println!(
        "{} replies, {n} words: median {median:.3}s, p90 {p90:.3}s, p99 {p99:.3}s",
        recordings.len()
    );
    assert!(n > 500, "only {n} words compared");
    assert!(
        median < LIMITS.0 && p90 < LIMITS.1 && p99 < LIMITS.2,
        "shown vs said over {n} words: median {median:.3}s, p90 {p90:.3}s, p99 {p99:.3}s"
    );
}

/// The checked-in replies' median, p90 and p99, in seconds: measured at 0.069, 0.260 and 0.885
/// (793 words), with about a third more room.
#[cfg(feature = "ffmpeg")]
const LIMITS: (f64, f64, f64) = (0.09, 0.35, 1.15);

/// `make bench-sync`: the accuracy over a recorded corpus (`LK_SYNC_CORPUS`), by provider. Each
/// run appends its overall numbers to the corpus's `history.tsv` under `LK_SYNC_COMMIT`, so the
/// drift across commits reads from one file.
#[test]
#[ignore = "needs a recorded corpus: LK_SYNC_CORPUS"]
#[allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a measurement over recorded files, not a decision"
)]
fn transcript_sync_accuracy() {
    use std::io::Write;
    let corpus = std::path::PathBuf::from(std::env::var("LK_SYNC_CORPUS").expect("LK_SYNC_CORPUS"));
    let mut dirs: Vec<_> =
        std::fs::read_dir(&corpus).expect("corpus").flatten().map(|d| d.path()).collect();
    dirs.sort();
    let mut by_provider: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    for r in dirs.iter().filter_map(|d| load(d)) {
        let provider = r.model.split('/').next().unwrap_or("").to_owned();
        let e = errors(&shown_tokens(&replay(&r)), &r.truth);
        by_provider.entry(provider).or_default().extend(e);
    }
    let all: Vec<f64> = by_provider.values().flatten().copied().collect();
    let mut line = String::new();
    for (name, e) in by_provider.into_iter().chain([("all".to_owned(), all)]) {
        let (median, p90, p99, n) = percentiles(e);
        println!("{name:12} {n:6} words  median {median:.3}s  p90 {p90:.3}s  p99 {p99:.3}s");
        line = format!("{n}\t{median:.3}\t{p90:.3}\t{p99:.3}");
    }
    let history = corpus.join("history.tsv");
    let previous = std::fs::read_to_string(&history).unwrap_or_default();
    if let Some(last) = previous.lines().skip(1).last() {
        println!("previous     {last}");
    }
    let commit = std::env::var("LK_SYNC_COMMIT").unwrap_or_else(|_| "unknown".to_owned());
    let mut file =
        std::fs::OpenOptions::new().create(true).append(true).open(&history).expect("history");
    if previous.is_empty() {
        writeln!(file, "commit\twords\tmedian\tp90\tp99").expect("history");
    }
    writeln!(file, "{commit}\t{line}").expect("history");
}
