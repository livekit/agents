//! The driver, end to end, over `lk-agents-inference`'s `FakeWs`.
//!
//! # Why there is no cassette here
//!
//! The plan offered a `wiremock` recording rig: proxy a real session once, replay it in CI. It was
//! not needed and would have been worse. `lk-agents-inference`'s [`WsTransport`] seam already
//! splits a socket into a scriptable sender and receiver, and its `FakeWs` records every frame the
//! client sent and lets a test feed replies *after* the client has acted -- which is the ordering
//! a cassette cannot express, because a cassette replays one recorded interleaving. Every
//! interesting realtime bug is an interleaving: a response that arrives after its request timed
//! out, a transcript revised after the agent started speaking, a delete acked with an empty id.
//! Those live in `machine.rs`'s tests, on a `VirtualClock`. This file's job is narrower and is the
//! only thing a cassette would have added: proof that the wiring between the machine, the codec
//! and the socket is connected the way the unit tests assume.
//!
//! Every await below is bounded by [`tokio::time::timeout`]; nothing here sleeps.

use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::TokioSpawner;
use lk_agents_abi::RealClock;
use lk_agents_inference::testkit::{FakeWs, NoImages, text};
use lk_agents_abi::Maybe;
use lk_agents_inference::realtime::{
    AudioDelta, Auth, Azure, Connect, Event, GenerationCreated, INFERENCE, NativeRealtime, OPENAI,
    OptionsPatch, SeqIds, XAI,
};
use serde_json::json;

/// No test may block for longer than this. A driver bug that parks forever fails as a timeout
/// rather than as a hung CI job.
const BOUND: Duration = Duration::from_secs(5);

async fn next_event(session: &lk_agents_inference::realtime::SessionHandle) -> Event {
    tokio::time::timeout(BOUND, session.next_event())
        .await
        .expect("the driver produced no event within the bound")
        .expect("the event stream ended early")
}

/// Wait until the fake has recorded at least `n` frames, or fail. Polling rather than sleeping:
/// the send is a `tokio` task hop away, not a delay.
async fn frames(ws: &FakeWs, n: usize) -> Vec<serde_json::Value> {
    tokio::time::timeout(BOUND, async {
        loop {
            let sent = ws.sent_json();
            if sent.len() >= n {
                return sent;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("expected {n} frames, saw {:?}", ws.sent()))
}

fn model(ws: Arc<FakeWs>, dialect: lk_agents_inference::realtime::Dialect) -> NativeRealtime {
    NativeRealtime::new(
        dialect,
        "sk-test",
        ws,
        Arc::new(TokioSpawner::current()),
        Arc::new(RealClock::new()),
        Arc::new(NoImages),
    )
    .with_ids(Arc::new(SeqIds::default()))
}

// Catches: the driver never sending the opening `session.update`, or sending it to a URL the
// provider does not serve. Both are silent -- the socket is open and the client looks healthy, but
// the server never learns the voice, the audio format or the turn-detection config, so it answers
// in its own defaults or not at all. The URL and the auth header are exactly the two things the
// unit tests cannot check, because neither reaches the machine.
#[tokio::test]
async fn opening_a_session_connects_and_configures_it() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let session = model(ws.clone(), OPENAI).session(false).await;

    let sent = frames(&ws, 1).await;
    assert_eq!(sent[0]["type"], json!("session.update"));
    assert_eq!(sent[0]["session"]["model"], json!("gpt-realtime"));
    assert_eq!(sent[0]["session"]["audio"]["output"]["voice"], json!("marin"));
    assert_eq!(sent[0]["session"]["audio"]["input"]["format"]["rate"], json!(24000));

    let (url, headers) = ws.last_connect().expect("connected");
    assert_eq!(url, "wss://api.openai.com/v1/realtime?model=gpt-realtime");
    assert!(headers.contains(&("Authorization".to_owned(), "Bearer sk-test".to_owned())));
    assert!(headers.contains(&("User-Agent".to_owned(), "LiveKit Agents".to_owned())));

    session.aclose().await;
}

// Catches: the xAI dialect being applied to the machine but not to the connection. Its base URL is
// already a `wss://` path, so a driver that re-derived one would connect to
// `wss://api.x.ai/v1/realtime/realtime`, and its `voice` must arrive at the top level of the
// session object or the server rejects the update. Both of these only exist once the dialect, the
// URL builder and the socket are wired together.
#[tokio::test]
async fn the_xai_dialect_reaches_both_the_url_and_the_payload() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let session = model(ws.clone(), XAI).session(false).await;

    let sent = frames(&ws, 1).await;
    assert_eq!(sent[0]["session"]["voice"], json!("Ara"));
    assert!(
        sent[0]["session"]["audio"]["output"].get("voice").is_none(),
        "promoted out of the nested block"
    );
    assert_eq!(sent[0]["session"]["turn_detection"]["type"], json!("server_vad"));
    assert_eq!(ws.last_connect().unwrap().0, "wss://api.x.ai/v1/realtime?model=grok-voice-latest");

    session.aclose().await;
}

// Catches: a broken loop between `generate_reply`, the wire and the event stream. The correlation
// is tested exhaustively in `machine.rs`; what is untested there is that the `req_id` the caller
// is handed is the one that actually goes out as `metadata.client_event_id`, and that the reply
// comes back out of the event channel rather than being dropped by the pump.
#[tokio::test]
async fn a_generate_reply_round_trips_through_the_socket() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let session = model(ws.clone(), OPENAI).session(false).await;
    frames(&ws, 1).await;

    let req_id = session.generate_reply(None);
    let sent = frames(&ws, 2).await;
    assert_eq!(sent[1]["type"], json!("response.create"));
    assert_eq!(sent[1]["response"]["metadata"]["client_event_id"], json!(req_id));

    ws.feed(text(json!({
        "type": "response.created",
        "response": {"id": "resp_1", "metadata": {"client_event_id": req_id}},
    })));

    loop {
        if let Event::GenerationCreated(GenerationCreated { response_id, req_id: got }) =
            next_event(&session).await
        {
            assert_eq!(response_id, "resp_1");
            assert_eq!(got, Some(req_id), "the caller's handle came back");
            break;
        }
    }
    session.aclose().await;
}

// Catches: audio deltas never reaching the consumer -- decoded but dropped, or emitted before the
// generation exists so the machine discards them. A session that configures itself correctly and
// then plays silence is the most expensive failure in this crate, and it is invisible to every
// pure test because it lives in the pump's dispatch.
#[tokio::test]
async fn server_audio_reaches_the_event_stream_as_frames() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let session = model(ws.clone(), OPENAI).session(false).await;
    frames(&ws, 1).await;

    // 4 bytes of base64 "AAAA" is 3 bytes of PCM; use a whole sample pair instead.
    let pcm = base64_of(&[1u8, 0, 2, 0]);
    ws.feed(text(json!({"type": "response.created", "response": {"id": "resp_1"}})));
    ws.feed(text(json!({
        "type": "response.output_item.added",
        "item": {"id": "msg_1", "type": "message"},
    })));
    ws.feed(text(json!({
        "type": "response.output_audio.delta", "item_id": "msg_1", "delta": pcm,
    })));

    loop {
        if let Event::AudioDelta (AudioDelta { message_id, frame, .. }) = next_event(&session).await {
            assert_eq!(message_id, "msg_1");
            assert_eq!(frame.sample_rate, 24_000);
            assert_eq!(frame.num_channels, 1);
            assert_eq!(&*frame.data, &[1u8, 0, 2, 0]);
            assert_eq!(frame.samples_per_channel, 2);
            break;
        }
    }
    session.aclose().await;
}

// Catches: `push_audio` bypassing the 100 ms re-framing, or base64-encoding the wrong buffer. The
// gateway accepts almost any chunk size, so a client that sent one frame per 10 ms `rtc` callback
// would work -- and would send ten times the frames, which is the kind of regression only a test
// that counts them catches.
#[tokio::test]
async fn pushed_audio_is_reframed_to_one_hundred_millisecond_chunks() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let session = model(ws.clone(), OPENAI).session(false).await;
    frames(&ws, 1).await;

    // 250 ms of 24 kHz mono i16, delivered as five 50 ms frames.
    for _ in 0..5 {
        session.push_audio(lk_agents_abi::AudioFrame {
            data: vec![0u8; 24_000 / 20 * 2].into(),
            sample_rate: 24_000,
            num_channels: 1,
            samples_per_channel: 24_000 / 20,
        });
    }
    let sent = frames(&ws, 3).await;
    let appends: Vec<_> =
        sent.iter().filter(|f| f["type"] == json!("input_audio_buffer.append")).collect();
    assert_eq!(appends.len(), 2, "250 ms is two whole 100 ms chunks, with 50 ms left buffered");

    session.send(lk_agents_abi::RealtimeCommand::CommitAudio(lk_agents_abi::llm::realtime::CommitAudio {}));
    let sent = frames(&ws, 4).await;
    assert_eq!(sent.last().unwrap()["type"], json!("input_audio_buffer.commit"));
    session.aclose().await;
}

fn base64_of(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

// Catches: the realtime handshake with no bound, or bounded on tokio's clock, which the session's
// virtual clock never moves -- so a gateway that accepts the socket and never finishes the
// handshake was a session that never produced an event. On the session clock it is the reference's
// `APIConnectionError("... connection timed out")`; with no retry budget it is final.
#[tokio::test]
async fn a_handshake_that_never_completes_is_reported_as_a_connection_timeout_on_the_session_clock()
{
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let ws = Arc::new(FakeWs::hanging());
    let mut connect = Connect::openai("sk-test", "https://api.openai.com/v1");
    connect.conn.max_retry = 0;
    let realtime = NativeRealtime::new(
        OPENAI,
        "sk-test",
        ws.clone(),
        Arc::new(TokioSpawner::current()),
        clock.clone(),
        Arc::new(NoImages),
    )
    .with_connect(connect)
    .with_ids(Arc::new(SeqIds::default()));
    let session = realtime.session(false).await;

    lk_agents_inference::testkit::arm_then_advance(&clock, Duration::from_secs(10)).await;
    match next_event(&session).await {
        Event::Error(err) => {
            assert!(err.message.contains("connection timed out"), "{}", err.message);
            assert!(!err.recoverable, "no retry budget: there is nothing to resume");
        }
        other => panic!("expected the connection timeout, got {other:?}"),
    }
    assert!(ws.sent().is_empty(), "nothing was sent on a socket that never opened");
}

// Catches: a dropped connection ending the session, or reconnecting without telling the caller
// or without replaying the session. The reference's `_main_task` reports the drop as a recoverable
// error, waits its first retry interval and reconnects, sending the session again.
#[tokio::test]
async fn an_unexpected_close_is_recoverable_and_the_session_reconnects() {
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    // the first socket closes at once (the script is empty and not kept open)
    let ws = Arc::new(FakeWs::new(vec![]));
    let realtime = NativeRealtime::new(
        OPENAI,
        "sk-test",
        ws.clone(),
        Arc::new(TokioSpawner::current()),
        clock.clone(),
        Arc::new(NoImages),
    )
    .with_ids(Arc::new(SeqIds::default()));
    let session = realtime.session(false).await;

    loop {
        match next_event(&session).await {
            Event::ConnectionAcquired { .. } | Event::Metrics(_) => continue,
            Event::Error(err) => {
                assert!(err.message.contains("closed unexpectedly"), "{}", err.message);
                assert!(err.recoverable);
                break;
            }
            other => panic!("expected the drop, got {other:?}"),
        }
    }
    lk_agents_inference::testkit::arm_then_advance(&clock, Duration::from_millis(100)).await;
    tokio::time::timeout(BOUND, async {
        while ws.connect_count() < 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the session reconnected");
    let sent = frames(&ws, 2).await;
    assert_eq!(sent[1]["type"], json!("session.update"), "the reconnection replays the session");
    session.aclose().await;
}

// Catches: Azure reached with the OpenAI URL or auth. Azure takes the deployment as `model` and an
// `api-key` header; its beta event names must read as the GA ones.
#[tokio::test]
async fn azure_gets_its_url_and_header() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let connect = Connect {
        base_url: "https://res.openai.azure.com/openai".into(),
        azure: Some(Azure { deployment: Some("gpt-rt".into()) }),
        auth: Auth::Azure { api_key: Some("az-key".into()), entra_token: None },
        conn: Default::default(),
    };
    let session = model(ws.clone(), OPENAI).with_connect(connect).session(false).await;

    frames(&ws, 1).await;
    let (url, headers) = ws.last_connect().unwrap();
    assert_eq!(url, "wss://res.openai.azure.com/openai/v1/realtime?model=gpt-rt");
    assert!(headers.contains(&("api-key".to_owned(), "az-key".to_owned())));
    assert!(!headers.iter().any(|(k, _)| k == "Authorization"));

    // a beta event name reads as the GA one
    ws.feed(text(json!({"type": "conversation.item.created", "previous_item_id": null,
        "item": {"id": "i1", "type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "hi"}]}})));
    loop {
        if let Event::RemoteItemAdded(added) = next_event(&session).await {
            assert_eq!(added.item.id(), "i1");
            break;
        }
    }
    session.aclose().await;
}

// Catches: LiveKit Inference reached with a static key instead of a gateway JWT, without the
// provider it routes to, or with `session.model` (the gateway picks the model from the URL).
#[tokio::test]
async fn livekit_inference_signs_each_connection_and_drops_the_session_model() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let connect = Connect {
        base_url: "https://agent-gateway.livekit.cloud/v1".into(),
        azure: None,
        auth: Auth::Inference {
            api_key: "APIkey".into(),
            api_secret: "secret".into(),
            provider: Some("openai".into()),
            inference_class: None,
            identity: Default::default(),
        },
        conn: Default::default(),
    };
    let realtime = model(ws.clone(), INFERENCE).with_connect(connect);
    let session = realtime.session(false).await;

    let sent = frames(&ws, 1).await;
    assert!(sent[0]["session"].get("model").is_none(), "{}", sent[0]);
    let (url, headers) = ws.last_connect().unwrap();
    assert_eq!(url, "wss://agent-gateway.livekit.cloud/v1/realtime?model=gpt-realtime");
    let auth = headers.iter().find(|(k, _)| k == "Authorization").expect("a bearer").1.clone();
    assert!(auth.starts_with("Bearer ey"), "a JWT, not the key: {auth}");
    assert!(headers.contains(&("X-LiveKit-Inference-Provider".to_owned(), "openai".to_owned())));
    session.aclose().await;
}

// Catches: `update_options` reaching the model but not its live sessions, or sending what did not
// change. The reference diffs each option against the session's own and sends one
// `options_update_` session.update with only the changes, the audio halves together.
#[tokio::test]
async fn update_options_sends_only_what_changed_to_the_live_session() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let realtime = model(ws.clone(), OPENAI);
    let session = realtime.session(false).await;
    frames(&ws, 1).await;

    realtime.update_options(OptionsPatch {
        voice: Maybe::Given("cedar".into()),
        speed: Maybe::Given(1.0),
        ..Default::default()
    });
    let sent = frames(&ws, 2).await;
    assert_eq!(sent[1]["type"], json!("session.update"));
    assert!(sent[1]["event_id"].as_str().unwrap().starts_with("options_update_"));
    assert_eq!(
        sent[1]["session"],
        json!({"type": "realtime", "audio": {"input": {}, "output": {"voice": "cedar"}}}),
        "speed did not change, transcription and noise reduction are the model's own"
    );
    assert_eq!(realtime.model(), "gpt-realtime");
    session.aclose().await;
}
