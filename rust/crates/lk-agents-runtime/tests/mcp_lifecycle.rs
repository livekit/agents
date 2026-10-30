//! Run with MCP_TEST_PYTHON pointing at Python with the MCP SDK and uvicorn installed.
use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::llm::mcp::{McpError, McpSession as _};
use lk_agents_net::env::ProcessEnv;
use lk_agents_runtime::mcp::McpSession;
use serde_json::Map;

type Result<T = ()> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

struct Peer {
    directory: PathBuf,
    python: String,
    script: String,
}
impl Peer {
    fn new(mode: &str) -> Self {
        let directory = std::env::temp_dir().join(format!("lk-mcp-{}-{mode}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        Self {
            directory,
            python: std::env::var("MCP_TEST_PYTHON").unwrap_or_else(|_| "python3".into()),
            script: Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/mcp/server.py")
                .display()
                .to_string(),
        }
    }
    fn args(&self, mode: &str) -> Vec<String> {
        vec![self.script.clone(), mode.into(), self.directory.display().to_string()]
    }
    async fn receipt(&self, name: &str) -> Result<String> {
        Ok(tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                match std::fs::read_to_string(self.directory.join(name)) {
                    Ok(text) => return Ok::<_, std::io::Error>(text),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        tokio::time::sleep(Duration::from_millis(10)).await
                    }
                    Err(error) => return Err(error),
                }
            }
        })
        .await??)
    }
}
impl Drop for Peer {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

async fn exercise(session: Arc<McpSession>, peer: &Peer) -> Result {
    let tools = session.list_tools().await?;
    assert_eq!(tools.as_array().unwrap().len(), 2, "must discover both pages");
    assert_eq!(tools[1]["name"], "second");
    let error = session.call_tool("rpc_error".into(), Map::new(), None, None).await.unwrap_err();
    match &error {
        McpError::Protocol { code, data, cause, .. } => {
            assert_eq!(*code, -32602);
            assert_eq!(data.as_ref().unwrap()["field"], "seat");
            assert!(cause.downcast_ref::<rmcp::ServiceError>().is_some());
        }
        other => panic!("lost MCP error taxonomy: {other:?}"),
    }
    let caller = session.clone();
    let call =
        tokio::spawn(async move { caller.call_tool("wait".into(), Map::new(), None, None).await });
    peer.receipt("called").await?;
    session.close().await;
    assert!(
        matches!(call.await?, Err(McpError::Cancelled(_))),
        "borrowed request must settle on close"
    );
    assert!(!session.connected());
    Ok(())
}

#[tokio::test]
async fn mcp_real_server_lifecycle_stdio() -> Result {
    let peer = Peer::new("stdio");
    let session = Arc::new(McpSession::new(30.0)?);
    let result = async {
        session.connect_stdio(peer.python.clone(), peer.args("stdio"), Vec::new(), None).await?;
        exercise(session.clone(), &peer).await?;
        let pid: i32 = peer.receipt("pid").await?.parse()?;
        #[cfg(unix)]
        assert!(
            !tokio::process::Command::new("kill")
                .args(["-0", &pid.to_string()])
                .output()
                .await?
                .status
                .success()
        );
        Ok(())
    }
    .await;
    session.close().await;
    result
}

#[tokio::test]
async fn mcp_real_server_lifecycle_http() -> Result {
    let peer = Peer::new("http");
    let mut child = tokio::process::Command::new(&peer.python)
        .args(peer.args("http"))
        .kill_on_drop(true)
        .spawn()?;
    let session = Arc::new(McpSession::new(30.0)?);
    let result = async {
        let url = peer.receipt("url").await?;
        session
            .connect_http(
                &ProcessEnv::current_runtime(),
                url,
                Vec::new(),
                "streamable_http".into(),
                5.0,
                30.0,
            )
            .await?;
        exercise(session.clone(), &peer).await
    }
    .await;
    session.close().await;
    child.kill().await?;
    child.wait().await?;
    result
}
