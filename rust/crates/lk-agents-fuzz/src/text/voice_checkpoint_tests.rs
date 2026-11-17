//! An applied checkpoint survives the loss of its caller and the end of its speech.

use std::sync::Arc;

use lk_agents_abi::delegation::remote::RemoteAgentState;
use lk_agents_abi::session::snapshot::SessionSnapshot;
use lk_agents_abi::storage::context::ContextActivation as _;
use lk_agents_abi::storage::session::{PersistedSessionState as _, SessionBinding as _};
use lk_agents_abi::storage::voice::{VoiceHistory, VoiceStorage as _, VoiceStore as _};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::text_session::SessionStateSink as _;
use lk_agents_runtime::net::{clock::SimClock, signal};
use lk_agents_runtime::storage::session_binding::SessionBinding;
use lk_agents_runtime::storage::session_state::SessionState;
use lk_agents_runtime::storage::voice::VoiceStore;
use serde_json::json;

use crate::fakes::storage::text::Store;

#[tokio::test(start_paused = true)]
async fn a_speech_ending_after_its_checkpoint_caller_dies_keeps_the_applied_remote_state() {
    let database = Store::open(31, false, Arc::new(SimClock::new())).await;
    let voice = Arc::new(
        VoiceStore::bound(
            database.backend.clone(),
            database.thread_id.clone(),
            "desk".into(),
            None,
            Cx::NONE,
        )
        .await
        .unwrap(),
    );
    let activation = voice.activation();
    let initial = SessionSnapshot::new("conversation".into(), "call".into(), "desk".into());
    let initial = json!({"version":1, "kind":"conversation", "state":initial});
    activation.stage_session(initial).unwrap();
    let view = SessionState::new(None).unwrap();
    let binding = SessionBinding::new(activation.clone(), view).unwrap();
    let _sink = activation.attach_session_sink(voice.session_sink()).unwrap();
    let history = voice.acquire(1, VoiceHistory::default()).await.unwrap();
    voice.checkpoint(1, 1, "desk".into(), history).await.unwrap();
    let scope = binding.checkpoint_voice(VoiceHistory::default()).await.unwrap();
    let mut received = activation.session_state().unwrap();
    let mut remote = RemoteAgentState::new("https://booking.test".into());
    remote.default_context_id = Some("trip".into());
    received["state"]["remote_agents"]["booking"] = serde_json::to_value(remote).unwrap();
    binding.update(received.to_string()).unwrap();

    let (applied, release) = database.hold_next_receipt("UPDATE partitions SET state =");
    let background = database.backend.background();
    let (done, ended) = signal::once();
    let caller = background.tasks.scoped(&*background.spawner, "checkpoint caller", async move {
        done.resolve(scope.commit().await);
    });
    applied.wait().await.unwrap();
    drop(caller);
    assert!(ended.wait().await.is_err(), "the caller ended before the acknowledgement");
    release.resolve(());
    voice.aclose().await.unwrap();

    assert_eq!(
        activation.session_state().unwrap()["state"],
        received["state"],
        "ending the speech retains the applied checkpoint"
    );
    assert!(!database.held("desk").await, "the call releases its partition");
    binding.close();
    database.close().await;
}
