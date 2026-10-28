//! Real transport framing and allocation limits. Each test records only its own thread's allocations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::Duration;

use lk_agents_worker::ipc::{StartJob, ToChild, ToParent, WireError, wire};
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};

async fn pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let (connected, accepted) =
        tokio::join!(TcpStream::connect(listener.local_addr().unwrap()), listener.accept());
    (connected.unwrap(), accepted.unwrap().0)
}

struct RecordingAllocator;
thread_local! {
    static RECORD: Cell<Option<usize>> = const { Cell::new(None) };
}

// SAFETY: every allocation and deallocation is forwarded unchanged to the system allocator.
// The thread-local counter records sizes only and never retains or accesses an allocation.
unsafe impl GlobalAlloc for RecordingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        RECORD.with(|record| {
            if let Some(max) = record.get() {
                record.set(Some(max.max(layout.size())));
            }
        });
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        RECORD.with(|record| {
            if let Some(max) = record.get() {
                record.set(Some(max.max(size)));
            }
        });
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: RecordingAllocator = RecordingAllocator;

#[tokio::test]
async fn ipc_oversize_preflight_avoids_allocating_an_encoded_frame() {
    let (mut sender, mut peer) = pair().await;
    let oversized = ToChild::StartJob {
        request_id: 1,
        job: Box::new(StartJob {
            metadata: "x".repeat(wire::MAX_FRAME + 1),
            ..StartJob::default()
        }),
    };
    RECORD.with(|record| record.set(Some(0)));
    let result = wire::write(&mut sender, &oversized).await;
    let max_allocation = RECORD.with(|record| record.replace(None).unwrap());
    assert!(result.is_err(), "the oversized frame must be refused");
    assert!(max_allocation < wire::MAX_FRAME, "encoded frame allocated {max_allocation} bytes");
    assert!(
        tokio::time::timeout(Duration::from_millis(10), wire::read::<ToParent, _>(&mut peer))
            .await
            .is_err(),
        "no prefix or payload is written for refused output"
    );
}

#[tokio::test]
async fn ipc_inbound_prefix_limit_refuses_without_reading_or_allocating_the_payload() {
    let (mut sender, mut peer) = pair().await;
    sender.write_all(&u32::MAX.to_be_bytes()).await.unwrap();
    RECORD.with(|record| record.set(Some(0)));
    let result =
        tokio::time::timeout(Duration::from_millis(100), wire::read::<ToParent, _>(&mut peer))
            .await;
    let max_allocation = RECORD.with(|record| record.replace(None).unwrap());
    assert!(matches!(result, Ok(Err(WireError::TooLarge(n))) if n == u32::MAX as usize));
    assert!(max_allocation < wire::MAX_FRAME, "advertised frame allocated {max_allocation} bytes");
}

#[tokio::test]
async fn real_ipc_frame_fault_matrix() {
    let message = ToParent::Pong { sent_ms: 1 << 40 };
    let payload = wire::encode(&message);
    let mut frame = (payload.len() as u32).to_be_bytes().to_vec();
    frame.extend(&payload);
    for fragment in [1, 2, 3, frame.len()] {
        let (mut sender, mut peer) = pair().await;
        let sending = async {
            for part in frame.chunks(fragment) {
                sender.write_all(part).await.unwrap();
                tokio::task::yield_now().await;
            }
        };
        let reading = wire::read::<ToParent, _>(&mut peer);
        let ((), read) =
            tokio::time::timeout(Duration::from_secs(1), async { tokio::join!(sending, reading) })
                .await
                .unwrap();
        assert_eq!(read.unwrap(), message, "fragment={fragment}");
    }
    // Every prefix/body truncation point must preserve UnexpectedEof, distinct from boundary EOF.
    for cut in 0..frame.len() {
        let (mut sender, mut peer) = pair().await;
        sender.write_all(&frame[..cut]).await.unwrap();
        sender.shutdown().await.unwrap();
        let error =
            tokio::time::timeout(Duration::from_secs(1), wire::read::<ToParent, _>(&mut peer))
                .await
                .unwrap()
                .unwrap_err();
        if cut == 0 {
            assert!(matches!(error, WireError::Eof));
        } else {
            assert!(
                matches!(error, WireError::Io { kind: std::io::ErrorKind::UnexpectedEof, .. }),
                "cut={cut}: {error}"
            );
        }
    }
    let (mut sender, mut peer) = pair().await;
    sender.write_all(&[0, 0, 0, 1, 99]).await.unwrap();
    assert!(matches!(wire::read::<ToParent, _>(&mut peer).await, Err(WireError::Malformed(_))));
}

#[tokio::test]
async fn real_ipc_saturation_preserves_health_and_settles_slow_writes_on_close() {
    use lk_agents_worker::ipc::{FLUSH_TIMEOUT, Hello, IpcClient, PROTOCOL_VERSION};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let (mut parent, child) = pair().await;
    wire::write(
        &mut parent,
        &ToChild::Hello(Hello {
            version: PROTOCOL_VERSION,
            asyncio_debug: false,
            http_proxy: None,
            health_timeout_ms: Some(100),
        }),
    )
    .await
    .unwrap();
    let orphaned = Arc::new(AtomicUsize::new(0));
    let count = orphaned.clone();
    let (client, _) = IpcClient::connect_with_health_action(child, move || {
        count.fetch_add(1, Ordering::SeqCst);
    })
    .await
    .unwrap();
    for _ in 0..65 {
        wire::write(&mut parent, &ToChild::DumpStacks).await.unwrap();
    }
    tokio::time::timeout(Duration::from_secs(1), async {
        while orphaned.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    let mut received = 0;
    while client.recv().await.is_some() {
        received += 1;
    }
    assert_eq!(received, 64, "saturation retains only the bounded application inbox");
    assert!(matches!(
        client.wait_reader().await,
        Err(WireError::Io { kind: std::io::ErrorKind::TimedOut, .. })
    ));
    client.close().await;
    assert_eq!(Arc::strong_count(&orphaned), 1, "the health callback's host reference settled");

    let (mut parent, child) = pair().await;
    wire::write(
        &mut parent,
        &ToChild::Hello(Hello {
            version: PROTOCOL_VERSION,
            asyncio_debug: false,
            http_proxy: None,
            health_timeout_ms: None,
        }),
    )
    .await
    .unwrap();
    let (client, _) = IpcClient::connect(child).await.unwrap();
    let sending = async {
        for _ in 0..1000 {
            client.send(ToParent::Exiting { reason: "x".repeat(64 << 10) }).await?;
        }
        Ok::<_, lk_agents_worker::ipc::Closed>(())
    };
    tokio::pin!(sending);
    assert!(
        tokio::time::timeout(Duration::from_millis(20), &mut sending).await.is_err(),
        "the undrained peer must saturate the bounded writer"
    );
    tokio::time::timeout(FLUSH_TIMEOUT + Duration::from_secs(1), client.close()).await.unwrap();
    assert!(sending.await.is_err(), "abandoned queued writes settle when close ends the writer");
}
