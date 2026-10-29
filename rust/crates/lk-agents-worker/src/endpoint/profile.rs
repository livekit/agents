//! Explicit real-TCP endpoint capacity profile; kept out of the fast unit lane.

use lk_agents_abi::SpawnerExt;
use lk_agents_net::task::TokioSpawner;
use std::sync::atomic::{AtomicBool, Ordering};

use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

const MESSAGE_BYTES: usize = 16 << 10;
const MESSAGES: usize = 8;
const UPGRADE: &[u8] = b"GET /ws HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n";

struct SocketIo {
    read: Mutex<OwnedReadHalf>,
    write: Mutex<OwnedWriteHalf>,
}

#[async_trait]
impl StreamIo for SocketIo {
    async fn read(&self, max: usize) -> Result<Vec<u8>, String> {
        let mut bytes = vec![0; max];
        let count = self.read.lock().await.read(&mut bytes).await.map_err(|e| e.to_string())?;
        bytes.truncate(count);
        Ok(bytes)
    }
    async fn write(&self, bytes: &[u8]) -> Result<(), String> {
        self.write.lock().await.write_all(bytes).await.map_err(|e| e.to_string())
    }
    async fn finish(&self) {
        self.write.lock().await.shutdown().await.unwrap();
    }
    async fn reset(&self, _: Reset) {
        self.finish().await;
    }
    async fn stop(&self, _: Reset) {}
}

struct ProfileApp;

#[async_trait]
impl AsgiApp for ProfileApp {
    async fn call(&self, _: Scope, conn: Conn) -> Result<(), AppError> {
        assert_eq!(conn.receive().await, Received::WsConnect);
        conn.send(Sent::WsAccept { subprotocol: None, headers: vec![] }).await.unwrap();
        for _ in 0..MESSAGES {
            let Received::WsBytes(bytes) = conn.receive().await else {
                panic!("profile payload was not delivered");
            };
            assert_eq!(bytes.len(), MESSAGE_BYTES);
        }
        conn.send(Sent::WsBytes(vec![b'y'; MESSAGE_BYTES])).await.unwrap();
        conn.send(Sent::WsClose { code: 1000, reason: String::new() }).await.unwrap();
        Ok(())
    }
}

async fn batch(sessions: usize, registry: Arc<Registry>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let spawner = TokioSpawner::current();
    let accept = spawner.spawn(async move {
        let spawner = TokioSpawner::current();
        let mut tasks = Vec::new();
        for _ in 0..sessions {
            let (socket, _) = listener.accept().await.unwrap();
            let (read, write) = socket.into_split();
            let io = Arc::new(SocketIo { read: Mutex::new(read), write: Mutex::new(write) });
            let registry = registry.clone();
            tasks.push(spawner.spawn(async move {
                serve(io, RequestInfo::default(), &ProfileApp, &registry).await;
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
    });
    let mut clients = Vec::new();
    for _ in 0..sessions {
        clients.push(spawner.spawn(async move {
            let mut socket = TcpStream::connect(address).await.unwrap();
            socket.set_nodelay(true).unwrap();
            socket.write_all(UPGRADE).await.unwrap();
            let mut frame = vec![0x82, 0xfe];
            frame.extend((MESSAGE_BYTES as u16).to_be_bytes());
            frame.extend([0; 4]);
            frame.extend(vec![b'x'; MESSAGE_BYTES]);
            for _ in 0..MESSAGES {
                socket.write_all(&frame).await.unwrap();
            }
            let mut response = Vec::new();
            socket.read_to_end(&mut response).await.unwrap();
            assert!(response.starts_with(b"HTTP/1.1 101"));
            assert!(response.len() >= MESSAGE_BYTES);
        }));
    }
    for client in clients {
        client.await.unwrap();
    }
    accept.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "explicit endpoint qualification profile over real TCP"]
async fn endpoint_resource_profile() {
    for sessions in [1, 8, 32, 64] {
        let registry = Arc::new(Registry::default());
        let running = Arc::new(AtomicBool::new(true));
        let monitor = {
            let running = running.clone();
            TokioSpawner::current().spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_millis(1));
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                let mut lags = Vec::new();
                while running.load(Ordering::Acquire) {
                    let due = interval.tick().await;
                    lags.push(
                        tokio::time::Instant::now().saturating_duration_since(due).as_micros(),
                    );
                }
                lags
            })
        };
        let start = std::time::Instant::now();
        let mut batches = 0;
        while start.elapsed() < Duration::from_secs(2) {
            tokio::time::timeout(Duration::from_secs(5), batch(sessions, registry.clone()))
                .await
                .unwrap();
            batches += 1;
        }
        running.store(false, Ordering::Release);
        let mut lags = monitor.await.unwrap();
        lags.sort_unstable();
        let p999 = lags[(lags.len() - 1) * 999 / 1000];
        let max = lags.last().unwrap();
        let elapsed = start.elapsed().as_secs_f64();
        let payload = sessions * batches * MESSAGE_BYTES * MESSAGES;
        let occupancy = registry.occupancy();
        assert_eq!(occupancy.handlers, 0);
        assert_eq!(occupancy.payload_bytes, 0);
        assert_eq!(occupancy.rejected_handlers, 0);
        assert_eq!(occupancy.rejected_payloads, 0);
        println!(
            "endpoint_profile sessions={sessions} batches={batches} bytes={payload} seconds={elapsed:.3} lag_p999_us={p999} lag_max_us={max} retained_handlers={} retained_payload_bytes={}",
            occupancy.handlers, occupancy.payload_bytes
        );
    }
}
