//! Task summaries through the real SDK HTTP adapter and the generated Storage contract.

use lk_agents_abi::storage::{Storage, StorageError, TaskState};
use lk_agents_runtime::storage::host::HostStorage;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

struct Peer {
    url: String,
    task: Option<JoinHandle<()>>,
}

impl Peer {
    async fn tasks() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            for (index, status) in ["input-required", "not-a-state"].into_iter().enumerate() {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let mut bytes = [0; 4096];
                while !request.ends_with(b"\r\n\r\n") {
                    let n = socket.read(&mut bytes).await.unwrap();
                    assert_ne!(n, 0);
                    request.extend_from_slice(&bytes[..n]);
                }
                let head = String::from_utf8(request).unwrap();
                if index == 0 {
                    assert!(head.starts_with("GET /v0/projects/p_/threads/AT_1/tasks?"));
                    assert!(head.contains("status=input-required"));
                }
                let body = serde_json::to_vec(&json!({
                    "items": [
                        {"taskId": "AST_1", "threadId": "AT_1", "status": status},
                        {"taskId": "AST_2", "threadId": "AT_1", "status": status, "name": ""}
                    ],
                    "pageInfo": {}
                }))
                .unwrap();
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len(),
                );
                socket.write_all(header.as_bytes()).await.unwrap();
                socket.write_all(&body).await.unwrap();
            }
        });
        Self { url, task: Some(task) }
    }

    async fn close(&mut self) {
        self.task.take().unwrap().await.unwrap();
    }
}

impl Drop for Peer {
    fn drop(&mut self) {
        if let Some(task) = &self.task {
            task.abort();
        }
    }
}

#[tokio::test]
async fn cloud_task_views_preserve_waiting_state_and_label_presence_and_refuse_unknown_states() {
    let mut peer = Peer::tasks().await;
    let storage = HostStorage::new(
        None,
        Some(peer.url.clone()),
        Some("key".into()),
        Some("secret".into()),
        None,
        None,
    )
    .unwrap();
    let tasks =
        storage.get_thread_tasks("AT_1".into(), Some(TaskState::InputRequired)).await.unwrap();
    assert_eq!(tasks.len(), 2);
    assert_eq!(tasks[0].status, TaskState::InputRequired);
    assert_eq!(tasks[0].name, None);
    assert_eq!(tasks[1].name.as_deref(), Some(""));
    assert!(matches!(
        storage.get_thread_tasks("AT_1".into(), None).await,
        Err(StorageError::Unavailable(_))
    ));
    storage.aclose().await;
    peer.close().await;
}
