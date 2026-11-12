use super::*;
use lk_agents_abi::a2a::door::TextEndpoint;
use lk_agents_abi::worker::server_host::ServerConfiguration;
use parking_lot::Mutex;

#[derive(Default)]
struct Registry(Mutex<Vec<Arc<dyn TextEndpoint>>>);
impl ServerConfiguration for Registry {
    fn text_endpoint(&self, name: &str) -> Option<Arc<dyn TextEndpoint>> {
        self.0.lock().iter().find(|door| door.name() == name).cloned()
    }
}

struct RoutedHost {
    label: &'static str,
    calls: Arc<Mutex<Vec<String>>>,
}
impl TurnHost for RoutedHost {
    fn handle<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        ctx: &'a TurnContext,
        _turn: Arc<dyn HandlerPort>,
        _cancel: Arc<dyn lk_agents_abi::invoke::Cancellation>,
    ) -> lk_agents_abi::BoxFuture<'a, Result<HandlerOutput, ProviderError>> {
        Box::pin(async move {
            assert_eq!(ctx.caller_history.is_some(), self.label == "beta");
            self.calls.lock().push(self.label.into());
            Ok(HandlerOutput { output: Some(json!(self.label)), ..Default::default() })
        })
    }
}

#[tokio::test]
async fn a_new_arrival_recovers_the_oldest_endpoint_and_its_caller_history() {
    use lk_agents_storage::thread::db::ReceiveInput;
    use lk_agents_storage::thread::schema::TaskData;
    let store = Backend::Sqlite.store(Default::default()).await;
    let created = store
        .storage
        .create_thread(livekit_protocol::agent::agent_thread::CreateRequest {
            scope: "customer:42".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    let thread = store.storage.open(&created.thread_id).await.unwrap();
    let holder = lk_agents_storage::Holder::new("old-worker");
    let mut held = thread
        .partition("desk")
        .open(lk_agents_storage::RunOptions::new("hold", ""), &holder)
        .await
        .unwrap();
    let mut message: a2a::Message = serde_json::from_value(json!({"messageId":"old-beta","contextId":created.thread_id,"role":"ROLE_USER","parts":[{"text":"old"}]})).unwrap();
    let caller = lk_agents_core::a2a::caller_history::CallerHistory::from_chat_items(
        &[lk_agents_abi::ChatItem::Message(lk_agents_abi::ChatMessage::text(
            "private",
            lk_agents_abi::Role::User,
            "caller history",
            std::time::UNIX_EPOCH,
        ))],
        None,
        &lk_agents_core::a2a::caller_history::HistorySharing::default(),
    )
    .unwrap();
    message.parts.push(caller.to_part().unwrap());
    let data = TaskData { history: vec![message.clone()], ..Default::default() };
    let (_, lease) = thread
        .partition("desk")
        .receive_at(
            &lk_agents_storage::Holder::new("dead-beta"),
            ReceiveInput { task_id: "AST_beta", message: &message, data: &data, internal: None },
            Some("beta"),
        )
        .await
        .unwrap();
    assert!(lease.is_none());
    held.commit(lk_agents_storage::CommitOptions::default()).await.unwrap();

    let registry = Arc::new(Registry::default());
    let calls = Arc::new(Mutex::new(Vec::new()));
    let env = lk_agents_net::env::ProcessEnv::current_runtime();
    let make = |label: &'static str| {
        let mut config = config();
        config.endpoint.name = label.into();
        config.options.policy = PolicySpec::All(Policy::Batch);
        Arc::new(
            Door::new(
                &env,
                store.namespace.clone(),
                config,
                Arc::new(RoutedHost { label, calls: calls.clone() }),
                Some(registry.clone()),
            )
            .unwrap(),
        )
    };
    let alpha = make("alpha");
    let beta = make("beta");
    registry.0.lock().extend([alpha.clone() as Arc<dyn TextEndpoint>, beta.clone()]);
    let response = alpha.dispatch(body("SendMessage", json!({"message":{"messageId":"new-alpha","contextId":created.thread_id,"role":"ROLE_USER","parts":[{"text":"new"}]}})), principal(true), None, None, None, None, None).await;
    while response.next().await.unwrap().is_some() {}
    assert_eq!(*calls.lock(), vec!["beta", "alpha"]);
    assert_eq!(
        thread.partition("desk").task("AST_beta").await.unwrap().unwrap().status,
        lk_agents_abi::storage::TaskState::Completed
    );
    assert!(thread.partition("desk").inbox(false).await.unwrap().is_empty());
    // Cleanup keeps the authoritative task receipt: a retry must not recreate inbox work.
    let retry = thread
        .partition("desk")
        .receive_at(
            &lk_agents_storage::Holder::new("retry"),
            ReceiveInput { task_id: "AST_retry", message: &message, data: &data, internal: None },
            Some("beta"),
        )
        .await
        .unwrap();
    assert!(
        matches!(retry.0, lk_agents_storage::thread::db::Receive::Duplicate(Some(ref id)) if id == "AST_beta")
    );
    assert!(retry.1.is_none());
    assert!(thread.partition("desk").inbox(false).await.unwrap().is_empty());
    alpha.aclose().await;
    beta.aclose().await;
    thread.close().await;
    store.close().await;
}
