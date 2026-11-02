//! The unmodified SDK thread client exercises the generated local HTTP contract.

use livekit_api::services::threads::ThreadClient;
use livekit_protocol::agent::{
    ChatMessage, ChatRole, agent_task, agent_thread, chat_context::chat_item,
};
use lk_agents_abi::a2a::auth::{AuthGrant, Credentials as _};
use lk_agents_abi::storage::http::ThreadServer as _;
use lk_agents_net::a2a_auth::Credentials;
use lk_agents_storage::http::{ThreadServer, ThreadService};
use lk_agents_storage::testing::Backend;
use lk_agents_storage::thread::run::TaskPatch;
use lk_agents_storage::{CommitOptions, ItemRow, RunOptions, TaskState};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::{Duration, timeout};

const KEY: &str = "local-key";
const SECRET: &str = "local-secret-local-secret-local-secret";

fn token(scope: &str) -> String {
    Credentials::new(KEY.into(), SECRET.into())
        .thread_grant_token(
            AuthGrant {
                scope: scope.into(),
                create: true,
                read: true,
                write: true,
                list: true,
                delete: true,
                ..Default::default()
            },
            "user".into(),
            3600,
        )
        .unwrap()
}

#[tokio::test]
async fn sdk_client_creates_updates_lists_and_deletes_threads() {
    let store = Backend::Sqlite.store(Default::default()).await;
    let server = ThreadServer::start(
        store.storage.clone(),
        "project-a".into(),
        KEY.into(),
        SECRET.into(),
        0,
    )
    .await
    .unwrap();
    let client = ThreadClient::with_token(
        &format!("http://127.0.0.1:{}", server.port()),
        &token("customer"),
    );
    let first = client
        .create(agent_thread::CreateRequest { scope: "customer".into(), ..Default::default() })
        .await
        .unwrap();
    assert_eq!(first.idle_ttl.unwrap().seconds, 86400);
    let second = client
        .create(agent_thread::CreateRequest { scope: "customer".into(), ..Default::default() })
        .await
        .unwrap();
    let page = client
        .list(agent_thread::ListRequest {
            scope: "customer".into(),
            limit: 1,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(page.threads.len(), 1);
    assert!(!page.next_page_token.is_empty());
    let next = client
        .list(agent_thread::ListRequest {
            scope: "customer".into(),
            limit: 1,
            page_token: page.next_page_token,
        })
        .await
        .unwrap();
    assert_eq!(next.threads.len(), 1);
    assert_ne!(next.threads[0].thread_id, page.threads[0].thread_id);
    let updated = client
        .update(agent_thread::UpdateRequest {
            thread_id: first.thread_id.clone(),
            attributes: [("title".into(), "hello".into())].into(),
            idle_ttl: Some(pbjson_types::Duration { seconds: 3600, nanos: 0 }),
        })
        .await
        .unwrap();
    assert_eq!(updated.attributes["title"], "hello");
    assert_eq!(client.get(&first.thread_id).await.unwrap().attributes["title"], "hello");
    client.delete(&first.thread_id).await.unwrap();
    assert!(client.get(&first.thread_id).await.is_err());
    client.delete(&second.thread_id).await.unwrap();
    server.aclose().await;
    server.aclose().await;
    store.close().await;
}

#[tokio::test]
async fn items_and_tasks_page_through_the_existing_client() {
    let store = Backend::Sqlite.store(Default::default()).await;
    let record = store
        .storage
        .create_thread(agent_thread::CreateRequest {
            scope: "customer".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    let thread = store.storage.open(&record.thread_id, "writer").await.unwrap();
    for index in 0..3 {
        let mut run =
            thread.lane("desk").open(RunOptions::new(format!("run-{index}"), "")).await.unwrap();
        run.history.push(ItemRow::new(chat_item::Item::Message(ChatMessage {
            id: format!("item-{index}"),
            role: ChatRole::User as i32,
            ..Default::default()
        })));
        run.tasks
            .write(
                &format!("task-{index}"),
                if index == 0 { TaskState::InputRequired } else { TaskState::Completed },
                TaskPatch::default(),
            )
            .await
            .unwrap();
        run.commit(CommitOptions::default()).await.unwrap();
    }
    thread.close().await;
    let server = ThreadServer::start(
        store.storage.clone(),
        "project-a".into(),
        KEY.into(),
        SECRET.into(),
        0,
    )
    .await
    .unwrap();
    let client = ThreadClient::with_token(
        &format!("http://127.0.0.1:{}", server.port()),
        &token("customer"),
    );
    let mut cursor = String::new();
    let mut seq = Vec::new();
    loop {
        let page = client
            .items(agent_thread::GetItemsRequest {
                thread_id: record.thread_id.clone(),
                agent: "desk".into(),
                limit: 1,
                page_token: cursor,
                sort_order: agent_thread::SortOrder::Desc as i32,
            })
            .await
            .unwrap();
        seq.extend(page.items.into_iter().map(|item| item.seq));
        cursor = page.next_page_token;
        if cursor.is_empty() {
            break;
        }
    }
    assert_eq!(seq, [3, 2, 1]);
    let page = client
        .tasks(agent_task::GetTasksRequest {
            thread_id: record.thread_id.clone(),
            status: Some("INPUT_REQUIRED".into()),
            limit: 1,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(page.tasks.len(), 1);
    assert_eq!(page.tasks[0].task_id, "task-0");
    assert_eq!(page.tasks[0].status, "INPUT_REQUIRED");
    let mut cursor = String::new();
    let mut tasks = Vec::new();
    loop {
        let page = client
            .tasks(agent_task::GetTasksRequest {
                thread_id: record.thread_id.clone(),
                limit: 1,
                page_token: cursor,
                ..Default::default()
            })
            .await
            .unwrap();
        tasks.extend(page.tasks.into_iter().map(|task| task.task_id));
        cursor = page.next_page_token;
        if cursor.is_empty() {
            break;
        }
    }
    assert_eq!(tasks, ["task-0", "task-1", "task-2"]);
    server.aclose().await;
    store.close().await;
}

#[tokio::test]
async fn authorization_validation_and_project_isolation_apply_before_mutation() {
    let store = Backend::Sqlite.store(Default::default()).await;
    let service =
        ThreadService::new(store.storage.clone(), "project-a".into(), KEY.into(), SECRET.into());
    let auth = format!("Bearer {}", token("customer"));
    let path = "/v0/projects/project-a/threads";
    assert_eq!(service.request("POST", path, "", None, b"{}").await.unwrap_err().status, 401);
    assert_eq!(
        service
            .request(
                "POST",
                "/v0/projects/project-b/threads",
                "",
                Some(&auth),
                br#"{"scope":"customer"}"#
            )
            .await
            .unwrap_err()
            .status,
        403
    );
    assert_eq!(
        service
            .request("POST", path, "", Some(&auth), br#"{"scope":"other"}"#)
            .await
            .unwrap_err()
            .status,
        403
    );
    for body in [
        br#"{"scope":"customer","unknown":true}"#.as_slice(),
        br#"{"scope":"customer","idleTtl":"P1D"}"#,
        br#"{"scope":"customer","idleTtl":"-1s"}"#,
        b"not JSON",
    ] {
        assert_eq!(
            service.request("POST", path, "", Some(&auth), body).await.unwrap_err().status,
            400
        );
    }
    let created =
        service.request("POST", path, "", Some(&auth), br#"{"scope":"customer"}"#).await.unwrap();
    let id = created["thread"]["threadId"].as_str().unwrap();
    assert_eq!(
        service
            .request("GET", path, "scope=customer&page.pageSize=bad", Some(&auth), b"")
            .await
            .unwrap_err()
            .status,
        400
    );
    let readonly = Credentials::new(KEY.into(), SECRET.into())
        .thread_grant_token(
            AuthGrant { scope: "customer".into(), read: true, ..Default::default() },
            "".into(),
            3600,
        )
        .unwrap();
    let readonly = format!("Bearer {readonly}");
    assert_eq!(
        service
            .request("DELETE", &format!("{path}/{id}"), "", Some(&readonly), b"")
            .await
            .unwrap_err()
            .status,
        403
    );
    let endpoint =
        Credentials::new(KEY.into(), SECRET.into()).project_token("".into(), 3600).unwrap();
    assert_eq!(
        service
            .request("GET", &format!("{path}/{id}"), "", Some(&format!("Bearer {endpoint}")), b"")
            .await
            .unwrap_err()
            .status,
        403
    );
    assert_eq!(
        service
            .request("PATCH", &format!("{path}/{id}"), "", Some(&auth), br#"{"idleTtl":"0s"}"#)
            .await
            .unwrap_err()
            .status,
        400
    );
    assert_eq!(
        service.request("GET", path, "scope=customer", Some(&auth), b"").await.unwrap()["items"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    store.close().await;
}

#[tokio::test]
async fn task_filters_preserve_every_state_and_optional_name() {
    let store = Backend::Sqlite.store(Default::default()).await;
    let record = store
        .storage
        .create_thread(agent_thread::CreateRequest {
            scope: "customer".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    let thread = store.storage.open(&record.thread_id, "writer").await.unwrap();
    let mut run = thread.lane("desk").open(RunOptions::new("states", "")).await.unwrap();
    let states = [
        (TaskState::Submitted, "SUBMITTED"),
        (TaskState::Working, "WORKING"),
        (TaskState::InputRequired, "INPUT_REQUIRED"),
        (TaskState::Completed, "COMPLETED"),
        (TaskState::Failed, "FAILED"),
        (TaskState::Canceled, "CANCELED"),
        (TaskState::Rejected, "REJECTED"),
    ];
    for (index, (state, _)) in states.iter().enumerate() {
        let name = match index {
            0 => None,
            1 => Some(String::new()),
            _ => Some(format!("label-{index}")),
        };
        run.tasks
            .write(&format!("task-{index}"), *state, TaskPatch { name, ..Default::default() })
            .await
            .unwrap();
    }
    run.commit(CommitOptions::default()).await.unwrap();
    thread.close().await;

    let service =
        ThreadService::new(store.storage.clone(), "project-a".into(), KEY.into(), SECRET.into());
    let auth = format!("Bearer {}", token("customer"));
    let path = format!("/v0/projects/project-a/threads/{}/tasks", record.thread_id);
    for (index, (_, status)) in states.iter().enumerate() {
        let page = service
            .request("GET", &path, &format!("status={status}"), Some(&auth), b"")
            .await
            .unwrap();
        let tasks = page["items"].as_array().unwrap();
        assert_eq!(tasks.len(), 1, "filter {status}");
        assert_eq!(tasks[0]["taskId"], format!("task-{index}"));
        assert_eq!(tasks[0]["status"], *status);
        match index {
            0 => assert!(tasks[0]["name"].is_null()),
            1 => assert_eq!(tasks[0]["name"], ""),
            _ => assert_eq!(tasks[0]["name"], format!("label-{index}")),
        }
    }

    let page = service.request("GET", &path, "page.pageSize=1", Some(&auth), b"").await.unwrap();
    let cursor = page["pageInfo"]["nextCursor"].as_str().unwrap();
    let query = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("status", "COMPLETED")
        .append_pair("page.cursor", cursor)
        .finish();
    assert_eq!(
        service.request("GET", &path, &query, Some(&auth), b"").await.unwrap_err().status,
        400
    );

    for query in [
        "status=RUNNING",
        "status=unknown",
        "status=7",
        "status=FAILED&status=REJECTED",
        "page.cursor=broken",
        "page.pageSize=-1",
        "page.pageSize=2147483648",
    ] {
        assert_eq!(
            service.request("GET", &path, query, Some(&auth), b"").await.unwrap_err().status,
            400,
            "query {query}"
        );
    }
    store.close().await;
}

#[tokio::test]
async fn url_decoding_rejects_invalid_input_and_preserves_encoded_values() {
    let store = Backend::Sqlite.store(Default::default()).await;
    let service =
        ThreadService::new(store.storage.clone(), "project-a".into(), KEY.into(), SECRET.into());
    let auth = format!("Bearer {}", token("customer + café"));
    let path = "/v0/projects/project%2Da/threads";
    let created = service
        .request("POST", path, "", Some(&auth), "{\"scope\":\"customer + café\"}".as_bytes())
        .await
        .unwrap();
    let page = service
        .request("GET", path, "scope=customer+%2B+caf%C3%A9", Some(&auth), b"")
        .await
        .unwrap();
    assert_eq!(page["items"][0]["threadId"], created["thread"]["threadId"]);

    for (path, query) in [
        ("/v0/projects/%FF/threads", "scope=customer"),
        (path, "scope=%FF"),
        (path, "scope=customer&sc%6Fpe=customer"),
        (path, "scope=customer&unknown=true"),
    ] {
        assert_eq!(
            service.request("GET", path, query, Some(&auth), b"").await.unwrap_err().status,
            400
        );
    }
    store.close().await;
}

async fn exchange(port: u16, request: &str) -> String {
    timeout(Duration::from_secs(5), async {
        let mut socket = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        socket.write_all(request.as_bytes()).await.unwrap();
        let mut response = String::new();
        socket.read_to_string(&mut response).await.unwrap();
        response
    })
    .await
    .expect("HTTP exchange did not finish")
}

#[tokio::test]
async fn listener_enforces_cors_body_limits_and_closes_idle_connections() {
    let store = Backend::Sqlite.store(Default::default()).await;
    let server = ThreadServer::start(
        store.storage.clone(),
        "project-a".into(),
        KEY.into(),
        SECRET.into(),
        0,
    )
    .await
    .unwrap();
    let port = server.port();
    let preflight = exchange(
        port,
        concat!(
            "OPTIONS /v0/projects/project-a/threads HTTP/1.1\r\n",
            "Host: localhost\r\n",
            "Origin: http://localhost:3000\r\n",
            "Access-Control-Request-Method: POST\r\n",
            "Access-Control-Request-Headers: Authorization, Content-Type\r\n\r\n",
        ),
    )
    .await;
    assert!(preflight.starts_with("HTTP/1.1 200"));
    assert!(preflight.contains("access-control-allow-origin: http://localhost:3000\r\n"));
    assert!(
        preflight
            .contains("access-control-allow-headers: Authorization, Content-Type, Api-Version\r\n")
    );

    for origin in
        ["https://example.com", "http://localhost.example.com", "http://localhost@evil.com"]
    {
        let response = exchange(
            port,
            &format!("OPTIONS / HTTP/1.1\r\nHost: localhost\r\nOrigin: {origin}\r\n\r\n"),
        )
        .await;
        assert!(response.starts_with("HTTP/1.1 403"), "origin {origin}: {response}");
        assert!(!response.contains("access-control-allow-origin:"));
    }

    let body = " ".repeat(64 * 1024 + 1);
    let request = format!(
        concat!(
            "POST /v0/projects/project-a/threads HTTP/1.1\r\n",
            "Host: localhost\r\n",
            "Content-Length: {}\r\n\r\n{}",
        ),
        body.len(),
        body
    );
    let oversized = exchange(port, &request).await;
    assert!(oversized.starts_with("HTTP/1.1 413"));

    let mut idle = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    idle.write_all(b"GET / HTTP/1.1\r\n").await.unwrap();
    timeout(Duration::from_secs(5), server.aclose())
        .await
        .expect("shutdown waited for idle client");
    assert!(TcpStream::connect(("127.0.0.1", port)).await.is_err());
    store.close().await;
}
