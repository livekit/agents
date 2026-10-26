//! The fakes against the truth. Every recording in `corpus/recordings` (`corpus/record.py`: real
//! providers through LiveKit Inference) is replayed through the fake that models its provider,
//! and what the fake does is compared with what the provider did:
//!
//! * the fake sends no event kind the provider never sent, and misses none it did;
//! * each response (realtime) or utterance (STT) has a shape, its event kinds in order with runs
//!   collapsed, the recordings have too;
//! * the documented orderings hold (developers.openai.com, realtime server events);
//! * every delay lies within what the provider was recorded taking.
//!
//! The realtime server is replayed against the recorded client frames and the recorded VAD
//! verdicts; the recogniser model against the recorded speech spans.
use std::collections::{BTreeMap, BTreeSet};

use lk_agents_abi::Ts;
use lk_agents_abi::utils::vocabulary::Coverage;
use lk_agents_fuzz::corpus;
use lk_agents_fuzz::fakes::realtime::server::{Exo, NoHooks, Server};
use lk_agents_fuzz::fakes::vocabulary;
use lk_agents_fuzz::search::stt_kinds;
use serde_json::{Value, json};

fn recordings(prefix: &str) -> Vec<(String, Value)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus/recordings");
    let mut out: Vec<(String, Value)> = std::fs::read_dir(&dir)
        .expect("corpus/recordings")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with(prefix)))
        .map(|p| {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            let value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
            (name, value)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(!out.is_empty(), "no {prefix} recordings");
    out
}

fn at(secs: f64) -> Ts {
    Ts::from_nanos((secs.max(0.0) * 1e9) as u64)
}

fn ms_of(t: Ts) -> f64 {
    t.as_nanos() as f64 / 1e6
}

// ================================================================================================
// The realtime server
// ================================================================================================

/// The server events the docs list (GA names), which the fake may send beyond the corpus when a
/// case asks for the failures the corpus never caught.
const DOCUMENTED: &[&str] = &[
    "error",
    "session.created",
    "session.updated",
    "conversation.created",
    "conversation.item.added",
    "conversation.item.created",
    "conversation.item.done",
    "conversation.item.retrieved",
    "conversation.item.input_audio_transcription.completed",
    "conversation.item.input_audio_transcription.delta",
    "conversation.item.input_audio_transcription.segment",
    "conversation.item.input_audio_transcription.failed",
    "conversation.item.truncated",
    "conversation.item.deleted",
    "input_audio_buffer.committed",
    "input_audio_buffer.dtmf_event_received",
    "input_audio_buffer.cleared",
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.timeout_triggered",
    "output_audio_buffer.started",
    "output_audio_buffer.stopped",
    "output_audio_buffer.cleared",
    "response.created",
    "response.done",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.output_text.delta",
    "response.output_text.done",
    "response.output_audio_transcript.delta",
    "response.output_audio_transcript.done",
    "response.output_audio.delta",
    "response.output_audio.done",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "response.mcp_call_arguments.delta",
    "response.mcp_call_arguments.done",
    "response.mcp_call.in_progress",
    "response.mcp_call.completed",
    "response.mcp_call.failed",
    "mcp_list_tools.in_progress",
    "mcp_list_tools.completed",
    "mcp_list_tools.failed",
    "rate_limits.updated",
];

/// Kinds the model chooses per response, at the corpus's rate (`function_call_share`).
const CHOSEN: &[&str] =
    &["response.function_call_arguments.delta", "response.function_call_arguments.done"];

/// The dice each session is replayed with, beyond the first: enough for a rare kind to show.
const SEEDS: u64 = 24;

fn turn_detection(kind: &str) -> Value {
    match kind {
        "semantic_vad" => json!({"type": "semantic_vad", "eagerness": "medium", "create_response": true, "interrupt_response": true}),
        "server_vad" => json!({"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 500, "create_response": true, "interrupt_response": true}),
        _ => Value::Null,
    }
}

/// What happens to the server at an instant: a client frame, or its VAD hearing the person.
enum Step {
    /// A client frame, built when it is sent: a truncation names the item the server made.
    Client(String),
    /// The person's speech reaching the buffer: this many ms of audio.
    Audio(f64),
    Exo(Exo),
}

/// One recording's session, replayed through the fake server with these dice: its events,
/// `(ms, frame)`.
fn replay(rec: &Value, seed: u64) -> Vec<(f64, Value)> {
    let mut server = Server::new(seed);
    let mut hooks = NoHooks(0);
    let mut steps: Vec<(f64, u32, Step)> = Vec::new();
    for c in rec["client"].as_array().unwrap() {
        let ty = c["type"].as_str().unwrap().to_owned();
        steps.push((c["t"].as_f64().unwrap(), 1, Step::Client(ty)));
    }
    for s in rec["speech"].as_array().unwrap() {
        let (t0, t1) = (s["t0"].as_f64().unwrap(), s["t1"].as_f64().unwrap());
        steps.push((t0, 0, Step::Audio((t1 - t0) * 1000.0)));
    }
    for e in rec["events"].as_array().unwrap() {
        let t = e["t"].as_f64().unwrap();
        match e["type"].as_str().unwrap() {
            "input_audio_buffer.speech_started" => steps.push((t, 2, Step::Exo(Exo::SpeechStarted))),
            "input_audio_buffer.speech_stopped" => steps.push((t, 2, Step::Exo(Exo::SpeechStopped))),
            _ => {}
        }
    }
    steps.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    server.connect(at(0.0));
    let mut out: Vec<(f64, Value)> = Vec::new();
    let mut last_message: Option<String> = None;
    let mut last_call: Option<String> = None;
    let mut truncations = 0;
    let drain = |server: &mut Server, until: Ts, out: &mut Vec<(f64, Value)>,
                     last_message: &mut Option<String>, last_call: &mut Option<String>| {
        while let Some(due) = server.next_due().filter(|d| *d <= until) {
            while let Some(frame) = server.pop(due, &mut NoHooks(0)) {
                if frame["type"] == "response.output_item.added" {
                    match frame["item"]["type"].as_str() {
                        Some("message") => *last_message = frame["item"]["id"].as_str().map(str::to_owned),
                        Some("function_call") => *last_call = frame["item"]["call_id"].as_str().map(str::to_owned),
                        _ => {}
                    }
                }
                out.push((ms_of(due), frame));
            }
        }
    };
    let td = turn_detection(rec["turn_detection"].as_str().unwrap_or("none"));
    for (t, _, step) in steps {
        let now = at(t);
        drain(&mut server, now, &mut out, &mut last_message, &mut last_call);
        match step {
            Step::Audio(ms) => {
                let pcm = vec![0u8; (ms / 1000.0 * 24_000.0) as usize * 2];
                let audio = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, pcm);
                server.client(now, &json!({"type": "input_audio_buffer.append", "audio": audio}), &mut hooks);
            }
            Step::Exo(exo) => {
                server.exo(now, exo);
            }
            Step::Client(ty) => {
                let frame = match ty.as_str() {
                    "session.update" => json!({"type": "session.update", "event_id": "s", "session": {
                        "type": "realtime", "output_modalities": ["audio"],
                        "audio": {"input": {"transcription": {"model": "gpt-4o-mini-transcribe"}, "turn_detection": td.clone()}},
                        "tools": [{"type": "function", "name": "lookup_order"}],
                    }}),
                    "response.create" => json!({"type": "response.create", "event_id": "r", "response": {}}),
                    "conversation.item.truncate" => {
                        truncations += 1;
                        // As the recorder did: a truncation inside what played, then one past it.
                        let end = if truncations == 1 { 400 } else { 999_999 };
                        json!({"type": "conversation.item.truncate", "item_id": last_message.clone().unwrap_or_default(), "content_index": 0, "audio_end_ms": end})
                    }
                    "conversation.item.create" => json!({"type": "conversation.item.create", "event_id": "c", "item": {
                        "type": "function_call_output", "call_id": last_call.clone().unwrap_or_default(), "output": "{}"}}),
                    other => json!({"type": other}),
                };
                server.client(now, &frame, &mut hooks);
            }
        }
    }
    let end = rec["events"].as_array().unwrap().last().unwrap()["t"].as_f64().unwrap() + 30.0;
    drain(&mut server, at(end), &mut out, &mut last_message, &mut last_call);
    out
}

fn types_of(events: &[(f64, Value)]) -> BTreeSet<String> {
    events.iter().map(|(_, e)| e["type"].as_str().unwrap().to_owned()).collect()
}

/// Each response's events, in order, as tokens: deltas of either kind are one token, and runs
/// are collapsed, so the shape is the order of the response's parts.
fn response_shapes(events: &[Value]) -> BTreeSet<String> {
    let mut of_item: BTreeMap<String, String> = BTreeMap::new();
    let mut shapes: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    for e in events {
        let ty = e["type"].as_str().unwrap();
        let resp = e["response_id"]
            .as_str()
            .or_else(|| e["response"]["id"].as_str())
            .map(str::to_owned)
            .or_else(|| {
                let item = e["item"]["id"].as_str().or_else(|| e["item_id"].as_str())?;
                of_item.get(item).cloned()
            });
        let Some(resp) = resp else { continue };
        if ty == "response.output_item.added"
            && let Some(item) = e["item"]["id"].as_str()
        {
            of_item.insert(item.to_owned(), resp.clone());
        }
        let token = match ty {
            "response.output_audio.delta" | "response.output_audio_transcript.delta" => "delta".to_owned(),
            "response.output_item.added" | "response.output_item.done" | "conversation.item.added"
            | "conversation.item.done" => {
                format!("{ty}:{}", e["item"]["type"].as_str().unwrap_or("?"))
            }
            other => other.to_owned(),
        };
        let seq = shapes.entry(resp.clone()).or_insert_with(|| {
            order.push(resp.clone());
            Vec::new()
        });
        if seq.last() != Some(&token) {
            seq.push(token);
        }
    }
    shapes.into_values().map(|s| s.join(" > ")).collect()
}

/// The documented orderings, on one session's events.
fn check_documented_order(name: &str, events: &[(f64, Value)]) {
    assert_eq!(events.first().map(|(_, e)| e["type"].clone()), Some(json!("session.created")), "{name}: session.created first");
    let mut started: BTreeSet<String> = BTreeSet::new();
    let mut committed: BTreeSet<String> = BTreeSet::new();
    let mut created: BTreeSet<String> = BTreeSet::new();
    let mut done: BTreeSet<String> = BTreeSet::new();
    let mut items_open: BTreeSet<String> = BTreeSet::new();
    let mut parts_open: BTreeSet<String> = BTreeSet::new();
    for (t, e) in events {
        let ty = e["type"].as_str().unwrap();
        let item = e["item_id"].as_str().map(str::to_owned);
        let resp = e["response_id"].as_str().or_else(|| e["response"]["id"].as_str()).map(str::to_owned);
        if let Some(r) = &resp {
            assert!(!done.contains(r), "{name}: {ty} after its response.done at {t}");
            if ty != "response.created" {
                assert!(created.contains(r), "{name}: {ty} before its response.created at {t}");
            }
        }
        match ty {
            "input_audio_buffer.speech_started" => {
                started.insert(item.unwrap());
            }
            "input_audio_buffer.speech_stopped" => {
                assert!(started.contains(&item.unwrap()), "{name}: speech_stopped with no speech_started at {t}");
            }
            "input_audio_buffer.committed" => {
                committed.insert(item.unwrap());
            }
            "conversation.item.input_audio_transcription.completed"
            | "conversation.item.input_audio_transcription.delta" => {
                assert!(committed.contains(&item.unwrap()), "{name}: {ty} before its commit at {t}");
            }
            "response.created" => {
                created.insert(resp.unwrap());
            }
            "response.done" => {
                done.insert(resp.unwrap());
            }
            "response.output_item.added" => {
                items_open.insert(e["item"]["id"].as_str().unwrap().to_owned());
            }
            "response.output_item.done" => {
                assert!(items_open.remove(e["item"]["id"].as_str().unwrap()), "{name}: output_item.done unopened at {t}");
            }
            "response.content_part.added" => {
                parts_open.insert(item.unwrap());
            }
            "response.content_part.done" => {
                assert!(parts_open.remove(&item.unwrap()), "{name}: content_part.done unopened at {t}");
            }
            "response.output_audio.delta" | "response.output_audio_transcript.delta" => {
                assert!(parts_open.contains(&item.unwrap()), "{name}: {ty} outside its content part at {t}");
            }
            _ => {}
        }
    }
}

/// The delays the fake took, by the calibration quantity each is drawn from.
fn delays(events: &[(f64, Value)]) -> BTreeMap<&'static str, Vec<f64>> {
    let mut out: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    let mut created: BTreeMap<String, f64> = BTreeMap::new();
    let mut item_at: BTreeMap<String, f64> = BTreeMap::new();
    let mut audio_seen: BTreeSet<String> = BTreeSet::new();
    let mut committed: BTreeMap<String, f64> = BTreeMap::new();
    let mut started_at: Option<f64> = None;
    for (t, e) in events {
        match e["type"].as_str().unwrap() {
            "input_audio_buffer.speech_started" => started_at = Some(*t),
            "input_audio_buffer.committed" => {
                committed.insert(e["item_id"].as_str().unwrap().to_owned(), *t);
            }
            "response.created" => {
                created.insert(e["response"]["id"].as_str().unwrap().to_owned(), *t);
            }
            "response.output_item.added" if e["item"]["type"] == "message" => {
                let resp = e["response_id"].as_str().unwrap();
                item_at.insert(e["item"]["id"].as_str().unwrap().to_owned(), *t);
                out.entry("item_after_created").or_default().push(t - created[resp]);
            }
            "response.output_audio.delta" => {
                let item = e["item_id"].as_str().unwrap().to_owned();
                if audio_seen.insert(item.clone()) {
                    out.entry("first_audio_after_item").or_default().push(t - item_at[&item]);
                }
            }
            "conversation.item.input_audio_transcription.completed" => {
                if let Some(c) = committed.get(e["item_id"].as_str().unwrap()) {
                    out.entry("transcription_after_committed").or_default().push(t - c);
                }
            }
            "response.done" if e["response"]["status_details"]["reason"] == "turn_detected" => {
                if let Some(s) = started_at {
                    out.entry("cancel_after_speech_started").or_default().push(t - s);
                }
            }
            _ => {}
        }
    }
    out
}

// Catches: a fake realtime server that sends what the real one never does, misses what it does,
// shapes a response or orders its events otherwise, or takes delays the real one never took.
#[test]
fn the_realtime_server_behaves_as_recorded() {
    let recorded = recordings("realtime-");
    let corpus_types: BTreeSet<String> =
        recorded.iter().flat_map(|(_, r)| r["events"].as_array().unwrap().iter()).map(|e| e["type"].as_str().unwrap().to_owned()).collect();
    let corpus_shapes: BTreeSet<String> = recorded
        .iter()
        .flat_map(|(_, r)| response_shapes(r["events"].as_array().unwrap()))
        .collect();
    // A kind most sessions have is owed by every replay of a session that has it; a rare one
    // (in under a third of the sessions, as `rate_limits.updated` is), or one whose presence is
    // the model's choice per response (a tool call), by the replays together.
    let sessions_with = |t: &String| {
        recorded
            .iter()
            .filter(|(_, r)| r["events"].as_array().unwrap().iter().any(|e| e["type"] == t.as_str()))
            .count()
    };
    let rare: BTreeSet<String> = corpus_types
        .iter()
        .filter(|t| sessions_with(t) * 3 < recorded.len() || CHOSEN.contains(&t.as_str()))
        .cloned()
        .collect();
    let mut fake_types: BTreeSet<String> = BTreeSet::new();
    let mut fake_delays: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    for (name, rec) in &recorded {
        for seed in 1..SEEDS {
            fake_types.extend(types_of(&replay(rec, seed)));
        }
        let events = replay(rec, 0);
        let types = types_of(&events);
        // Never something the real server never sent.
        let unseen: Vec<&String> = types.difference(&corpus_types).collect();
        assert!(unseen.is_empty(), "{name}: the fake sent {unseen:?}, which no recording has");
        // Never missing something the real server sent in this session.
        let recorded_types: BTreeSet<String> = rec["events"].as_array().unwrap().iter().map(|e| e["type"].as_str().unwrap().to_owned()).collect();
        let missing: Vec<&String> =
            recorded_types.difference(&types).filter(|t| !rare.contains(*t)).collect();
        assert!(missing.is_empty(), "{name}: the real server sent {missing:?}, the fake never did");
        // Every response shaped as a real one was.
        let frames: Vec<Value> = events.iter().map(|(_, e)| e.clone()).collect();
        for shape in response_shapes(&frames) {
            assert!(corpus_shapes.contains(&shape), "{name}: a response shaped as no recorded one was:\n  {shape}\nrecorded:\n  {}", corpus_shapes.iter().cloned().collect::<Vec<_>>().join("\n  "));
        }
        check_documented_order(name, &events);
        for (k, v) in delays(&events) {
            fake_delays.entry(k).or_default().extend(v);
        }
        fake_types.extend(types);
    }
    // A rare kind, by the replays together.
    let missing: Vec<&String> = rare.difference(&fake_types).collect();
    assert!(missing.is_empty(), "the real server sent {missing:?} now and then, the fake never did");
    // Every delay within what the real server was recorded taking.
    for (quantity, values) in &fake_delays {
        let (lo, hi) = corpus::range(&["realtime", quantity]);
        for v in values {
            assert!(*v >= lo - 1.0 && *v <= hi + 1.0, "{quantity}: the fake took {v} ms, the recordings {lo}..{hi}");
        }
    }
    // The wire vocabulary claims: every server event the table says the fake sends, it sent in
    // the replays or is a documented event only a failure input makes.
    for (variant, coverage) in vocabulary::realtime_wire_claims() {
        if let Coverage::Covered(event) = coverage {
            assert!(DOCUMENTED.contains(&event), "{variant}: {event} is no documented server event");
            let failure_only = matches!(event, "conversation.item.deleted" | "conversation.item.input_audio_transcription.failed");
            assert!(fake_types.contains(event) || failure_only, "{variant}: the fake never sent {event} in the replays");
        }
    }
    for t in &corpus_types {
        assert!(DOCUMENTED.contains(&t.as_str()), "the corpus has {t}, which the docs list does not: update DOCUMENTED");
    }
}

// Catches: the server's failure inputs sending events outside the documented vocabulary.
#[test]
fn the_realtime_failures_send_documented_events() {
    let mut server = Server::new(3);
    let mut hooks = NoHooks(0);
    server.connect(at(0.0));
    let session = json!({"type": "session.update", "session": {"output_modalities": ["audio"], "audio": {"input": {
        "transcription": {"model": "m"}, "turn_detection": turn_detection("server_vad")}}}});
    server.client(at(0.0), &session, &mut hooks);
    let pcm = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, vec![1u8; 4800]);
    server.client(at(0.1), &json!({"type": "input_audio_buffer.append", "audio": pcm}), &mut hooks);
    assert!(server.exo(at(0.2), Exo::SpeechStarted));
    assert!(server.exo(at(1.0), Exo::SpeechStopped));
    assert!(server.exo(at(1.05), Exo::ResponseFailed));
    assert!(server.exo(at(1.1), Exo::TranscriptionFailed));
    assert!(server.exo(at(1.2), Exo::ServerError));
    assert!(server.exo(at(1.3), Exo::StrayDelta));
    server.client(at(2.0), &json!({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": []}}), &mut hooks);
    server.client(at(2.1), &json!({"type": "conversation.item.delete", "item_id": "nope"}), &mut hooks);
    let mut types = BTreeSet::new();
    while let Some(due) = server.next_due() {
        while let Some(frame) = server.pop(due, &mut hooks) {
            types.insert(frame["type"].as_str().unwrap().to_owned());
        }
    }
    for t in &types {
        assert!(DOCUMENTED.contains(&t.as_str()), "{t} is no documented server event");
    }
    assert!(types.contains("conversation.item.input_audio_transcription.failed"));
}

// ================================================================================================
// The recogniser
// ================================================================================================

/// Each utterance's event kinds, from its start of speech to its end, runs collapsed, and the
/// interim-preflight alternation Flux repeats as long as the person goes on folded into one
/// `(interim > preflight)+`: how many times is the length of the turn, not its shape.
fn utterance_shapes(kinds: &[(f64, String)]) -> BTreeSet<String> {
    let mut shapes = BTreeSet::new();
    let mut cur: Vec<String> = Vec::new();
    for (_, k) in kinds {
        if k == "start_of_speech" {
            cur.clear();
        }
        if cur.last() != Some(k) {
            cur.push(k.clone());
        }
        if k == "end_of_speech" {
            let mut folded: Vec<String> = Vec::new();
            let mut i = 0;
            while i < cur.len() {
                let pair = cur.get(i).map(String::as_str) == Some("interim_transcript")
                    && cur.get(i + 1).map(String::as_str) == Some("preflight_transcript");
                if pair {
                    while cur.get(i).map(String::as_str) == Some("interim_transcript")
                        && cur.get(i + 1).map(String::as_str) == Some("preflight_transcript")
                    {
                        i += 2;
                    }
                    folded.push("(interim_transcript > preflight_transcript)+".into());
                } else {
                    folded.push(cur[i].clone());
                    i += 1;
                }
            }
            shapes.insert(folded.join(" > "));
            cur.clear();
        }
    }
    shapes
}

// Catches: a recogniser model whose utterances are shaped as the recorded provider's never were
// (a start of speech before its first words, a final with no usage before it, an end of speech
// apart from its final, a preflight from a model that sends none) or whose delays lie outside what
// it was recorded taking.
#[test]
fn the_recogniser_behaves_as_recorded() {
    let mut by_model: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for (_, rec) in recordings("stt-") {
        by_model.entry(rec["model"].as_str().unwrap().to_owned()).or_default().push(rec);
    }
    for model in corpus::STT_MODELS {
        let recs = by_model.get(*model).unwrap_or_else(|| panic!("no recording of {model}"));
        let mut recorded = BTreeSet::new();
        for rec in recs {
            let kinds: Vec<(f64, String)> = rec["events"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| (e["t"].as_f64().unwrap(), e["type"].as_str().unwrap().to_owned()))
                .collect();
            recorded.extend(utterance_shapes(&kinds));
        }
        let (first_lo, first_hi) = corpus::range(&["stt", model, "first_after_onset"]);
        let (final_lo, final_hi) = corpus::range(&["stt", model, "final_after_end"]);
        let mut fake = BTreeSet::new();
        for rec in recs {
            for (i, span) in rec["speech"].as_array().unwrap().iter().enumerate() {
                let dur = ((span["t1"].as_f64().unwrap() - span["t0"].as_f64().unwrap()) * 1000.0) as u64;
                let words = "one two three four five six seven eight nine ten";
                for seed in 0..20 {
                    let kinds = stt_kinds(model, words, dur, seed * 31 + i as u64);
                    let as_owned: Vec<(f64, String)> = kinds.iter().map(|(t, k)| (*t as f64, (*k).to_owned())).collect();
                    fake.extend(utterance_shapes(&as_owned));
                    // The first words of the first piece, and the last final, within the recorded
                    // delays (the cuts of a long turn place later pieces past the first).
                    if let Some((t, _)) = kinds.iter().find(|(_, k)| *k == "interim_transcript" || *k == "final_transcript") {
                        let lag = *t as f64;
                        assert!(lag >= first_lo.min(dur as f64) - 1.0 && lag <= first_hi.max(dur as f64 + final_hi) + 1.0,
                            "{model}: first words {lag} ms after onset, recorded {first_lo}..{first_hi}");
                    }
                    if let Some((t, _)) = kinds.iter().rev().find(|(_, k)| *k == "final_transcript") {
                        let tail = *t as f64 - dur as f64;
                        assert!(tail <= final_hi + first_hi + 1.0 && tail >= final_lo.min(0.0) - 1.0,
                            "{model}: last final {tail} ms after the end, recorded {final_lo}..{final_hi}");
                    }
                }
            }
            // Noise, for a model that reports it.
            fake.extend(utterance_shapes(
                &stt_kinds(model, "", 400, 1).iter().map(|(t, k)| (*t as f64, (*k).to_owned())).collect::<Vec<_>>(),
            ));
        }
        for shape in &fake {
            assert!(recorded.contains(shape), "{model}: an utterance shaped as no recorded one was:\n  {shape}\nrecorded:\n  {}",
                recorded.iter().cloned().collect::<Vec<_>>().join("\n  "));
        }
        // And every recorded shape is one the model makes.
        for shape in &recorded {
            assert!(fake.contains(shape), "{model}: the provider shaped an utterance as\n  {shape}\nand the model never does; it makes:\n  {}",
                fake.iter().cloned().collect::<Vec<_>>().join("\n  "));
        }
    }
}
