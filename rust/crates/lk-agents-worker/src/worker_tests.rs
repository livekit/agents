//! Connection lifecycle and scheduling tests using scripted peers and gated host operations.

#[path = "worker_test_fixtures.rs"]
mod fixtures;
use fixtures::*;

use crate::transport::TransportError;
use crate::*;
use livekit_protocol::{self as protocol, ServerMessage, server_message};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;

// The two pure decisions

// Catches: an exponential backoff replacing `min(n * 2, 10)`, losing the immediate first retry.
#[test]
fn the_first_reconnect_is_immediate_and_the_ladder_caps_at_ten() {
    let ladder: Vec<u64> = (0..8).map(|n| retry_delay(n).as_secs()).collect();
    assert_eq!(ladder, vec![0, 2, 4, 6, 8, 10, 10, 10]);
}

// The connection itself

// Catches: the register message not being the first thing written to the socket.

#[tokio::test(start_paused = true)]
async fn register_is_the_first_message_written_and_the_worker_id_lands() {
    let (mut c, wire, _h) = conn(vec![Step::Serves(vec![register_message("W-7")])], 0);
    // The stream blocks after the register response, so drive one connection with a deadline.
    let _ = tokio::time::timeout(Duration::from_millis(50), c.run()).await;
    let sent = &wire.lock().sent;
    assert!(!sent.is_empty(), "nothing was written to the socket");
    let first = &sent[0];
    assert!(
        matches!(first.message, Some(livekit_protocol::worker_message::Message::Register(_))),
        "first message was {:?}",
        first.message
    );
    assert_eq!(c.snapshot_now().worker_id, "W-7");
}

// Catches: an off-by-one in the retry ladder; `max_retry = 2` means three attempts, not two.
#[tokio::test(start_paused = true)]
async fn max_retry_two_means_three_connect_attempts() {
    let (mut c, wire, _h) =
        conn(vec![Step::ConnectFails, Step::ConnectFails, Step::ConnectFails], 2);
    let err = c.run().await.unwrap_err();
    assert!(matches!(err, RuntimeError::RetriesExhausted { retries: 2, .. }), "got {err:?}");
    assert_eq!(wire.lock().attempts, 3);
}

// Catches: treating a mid-session close as fatal instead of reconnecting.

#[tokio::test(start_paused = true)]
async fn a_closed_connection_reconnects_and_the_machine_is_told() {
    let (mut c, wire, _h) = conn(
        vec![
            Step::Serves(vec![
                register_message("W-1"),
                Err(TransportError::Failed("peer disconnected".into())),
            ]),
            Step::Serves(vec![register_message("W-2")]),
        ],
        5,
    );
    let _ = tokio::time::timeout(Duration::from_millis(50), c.run()).await;
    assert_eq!(wire.lock().attempts, 2, "did not reconnect after the close");
    assert_eq!(c.snapshot_now().worker_id, "W-2", "did not re-register on the new socket");
}

// Catches: the reconnect budget spent over the worker's lifetime instead of per outage (the
// reference resets it on each connect): with `max_retry = 1`, an outage after a recovered one
// still gets its retry.

#[tokio::test(start_paused = true)]
async fn a_connection_restores_the_retry_budget() {
    let (mut c, wire, _h) = conn(
        vec![
            Step::ConnectFails,
            Step::Serves(vec![
                register_message("W-1"),
                Err(TransportError::Failed("peer disconnected".into())),
            ]),
            Step::Serves(vec![register_message("W-2")]),
        ],
        1,
    );
    let _ = tokio::time::timeout(Duration::from_secs(30), c.run()).await;
    assert_eq!(wire.lock().attempts, 3);
    assert_eq!(c.snapshot_now().worker_id, "W-2", "gave up after the second outage");
}

// Catches: `Drain` not reaching the machine while `run()` is in flight.

#[tokio::test(start_paused = true)]
async fn drain_reaches_a_running_connection_from_another_task() {
    let (mut c, _wire, h) = conn(vec![Step::Serves(vec![register_message("W-1")])], 0);
    let h2 = h.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(5)).await;
        h2.send(WorkerCommand::Drain { timeout: None }).unwrap();
    });
    let _ = tokio::time::timeout(Duration::from_millis(200), c.run()).await;
    assert!(c.snapshot_now().draining, "drain never reached the machine");
    assert!(h.snapshot().draining, "the snapshot did not follow the machine");
}

// Catches: `Close` being ignored during the reconnect backoff.

#[tokio::test(start_paused = true)]
async fn close_during_the_reconnect_backoff_stops_the_ladder() {
    // enough failures to exhaust patience if Close were ignored
    let steps = (0..20).map(|_| Step::ConnectFails).collect();
    let (mut c, wire, h) = conn(steps, 50);
    let h2 = h.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(3)).await;
        h2.send(WorkerCommand::Close).unwrap();
    });
    c.run().await.expect("close should be a clean return, not GaveUp");
    // 0s + 2s of backoff elapsed, so only a handful of attempts -- nowhere near 50
    assert!(wire.lock().attempts < 5, "kept reconnecting after Close");
    assert_eq!(h.snapshot().finished.as_deref(), Some("closed"));
}

// The idle-process target

// Catches: the idle-process target being computed and then dropped before it reaches the handle,
// a snapshot read stealing it from the next reader, or devmode's infinite threshold ("leave the
// pool alone") turned into a target of zero.

#[tokio::test(start_paused = true)]
async fn registering_leaves_the_idle_process_target_on_the_handle() {
    let prod = WorkerConfig { ws_url: "wss://gw".into(), ..Default::default() };
    let dev = WorkerConfig {
        devmode: true,
        load_threshold: EnvOption::Split { dev: f32::INFINITY, prod: 0.75 },
        ..prod.clone()
    };
    for (cfg, want) in [(prod, Some(4)), (dev, None)] {
        let steps = vec![Step::Serves(vec![register_message("W-1")])];
        let (mut c, _wire, h) = conn_with(steps, 0, cfg, LoadHost::default());
        let _ = tokio::time::timeout(Duration::from_millis(50), c.run()).await;
        assert_eq!(h.snapshot().target_idle_processes, want);
        assert_eq!(h.snapshot().target_idle_processes, want, "a read must not consume it");
    }
}

// The default load function: which side samples, and what "no answer" means

// Catches: the host inventing a load when there is no `load_fnc`; `None` must make the core sample.
#[tokio::test(start_paused = true)]
async fn no_user_load_function_means_the_core_samples_for_itself() {
    let (mut c, _h) = load_conn(LoadHost::default());
    c.sample_load_now().await.expect("sampled");
    c.sample_load_now().await.expect("sampled");
    // With no user override, use the core CPU monitor.
    let load = c.snapshot_now().worker_load;
    assert!(load.is_finite() && (0.0..=1.0).contains(&load), "load out of range: {load}");
}

// Catches: the user's `load_fnc` being ignored for the CPU reading, or asked twice per sample.

#[tokio::test(start_paused = true)]
async fn a_user_load_function_wins_and_is_asked_exactly_once() {
    let user_calls = Arc::new(Mutex::new(0));
    let (mut c, h) = load_conn(LoadHost { user: Some(0.42), user_calls: user_calls.clone() });
    c.sample_load_now().await.expect("sampled");
    assert!((c.snapshot_now().worker_load - 0.42).abs() < 1e-6, "{}", c.snapshot_now().worker_load);
    assert_eq!(*user_calls.lock(), 1);
    assert_eq!(h.snapshot().worker_load, c.snapshot_now().worker_load);
}

// Runtime scheduling tests use channels for the peer and gates for slow host operations.
#[tokio::test]
async fn acceptance_delivers_the_job_and_participant_fields_to_the_peer() {
    let RuntimeRig { mut runtime, handle, peer, mut sent, decisions, started: _started, .. } =
        runtime_rig();
    peer.send(register_message("worker")).unwrap();
    decisions.add_permits(1);
    let task = tokio::spawn(async move { runtime.run().await });
    peer.send(Ok(offer("accepted-job"))).unwrap();
    let message = until_sent(&mut sent, |message| {
        matches!(&message.message,
        Some(livekit_protocol::worker_message::Message::Availability(reply)) if reply.available)
    })
    .await;
    handle.send(WorkerCommand::Close).unwrap();
    task.await.unwrap().unwrap();
    let Some(livekit_protocol::worker_message::Message::Availability(reply)) = message.message
    else {
        unreachable!()
    };
    assert_eq!(reply.job_id, "accepted-job");
    assert_eq!(reply.participant_identity, "agent-identity");
    assert_eq!(reply.participant_name, "Agent Name");
    assert_eq!(reply.participant_metadata, "participant-metadata");
    assert_eq!(reply.participant_attributes[lk_agents_abi::ATTRIBUTE_AGENT_NAME], "per-job-agent");
}

#[tokio::test(start_paused = true)]
async fn slow_decisions_do_not_block_other_requests_status_timers_or_close() {
    let RuntimeRig { mut runtime, handle, peer, mut sent, mut started, dropped, .. } =
        runtime_rig();
    peer.send(register_message("worker")).unwrap();
    let task = tokio::spawn(async move { runtime.run().await });
    peer.send(Ok(offer("a"))).unwrap();
    peer.send(Ok(offer("b"))).unwrap();
    for _ in 0..2 {
        assert_eq!(started.recv().await, Some("decision"));
    }
    until_sent(&mut sent, |m| {
        matches!(m.message, Some(livekit_protocol::worker_message::Message::UpdateWorker(_)))
    })
    .await;
    assert_eq!(handle.snapshot().reserved_slots, 2);
    handle.send(WorkerCommand::Close).unwrap();
    tokio::time::timeout(Duration::from_secs(1), task).await.unwrap().unwrap().unwrap();
    assert_eq!(
        dropped.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "close must cancel outstanding host work"
    );
}

#[tokio::test(start_paused = true)]
async fn drain_timeout_is_reported_while_a_host_decision_is_pending() {
    let RuntimeRig { mut runtime, handle, peer, mut started, sent: _sent, .. } = runtime_rig();
    peer.send(register_message("worker")).unwrap();
    peer.send(Ok(offer("a"))).unwrap();
    let mut notifications = handle.subscribe();
    let task = tokio::spawn(async move { runtime.run().await });
    assert_eq!(started.recv().await, Some("decision"));
    assert_eq!(handle.drain(Some(Duration::from_secs(1))).await, Err(DrainError::TimedOut));
    assert_eq!(handle.snapshot().drain_outcome, Some(DrainOutcome::TimedOut));
    let mut observed = false;
    while let Ok(event) = notifications.try_recv() {
        observed |= matches!(event, WorkerNotification::Drained(DrainOutcome::TimedOut));
    }
    assert!(observed);
    handle.send(WorkerCommand::Close).unwrap();
    task.await.unwrap().unwrap();
}

#[tokio::test(start_paused = true)]
async fn slow_launch_allows_status_updates_and_drain_waits_for_early_job_completion() {
    let RuntimeRig {
        mut runtime, handle, peer, mut sent, mut started, decisions, launches, ..
    } = runtime_rig();
    decisions.add_permits(1);
    peer.send(register_message("worker")).unwrap();
    peer.send(Ok(offer("a"))).unwrap();
    let task = tokio::spawn(async move { runtime.run().await });
    assert_eq!(started.recv().await, Some("decision"));
    until_sent(&mut sent, |m| {
        matches!(m.message, Some(livekit_protocol::worker_message::Message::Availability(_)))
    })
    .await;
    peer.send(Ok(assignment_message("a"))).unwrap();
    assert_eq!(started.recv().await, Some("launch"));
    until_sent(&mut sent, |m| {
        matches!(m.message, Some(livekit_protocol::worker_message::Message::UpdateWorker(_)))
    })
    .await;
    handle
        .send(WorkerCommand::JobStatus {
            job_id: "a".into(),
            status: protocol::JobStatus::JsSuccess,
        })
        .unwrap();
    let h = handle.clone();
    let drain = tokio::spawn(async move { h.drain(Some(Duration::from_secs(5))).await });
    tokio::task::yield_now().await;
    assert!(!drain.is_finished(), "drain must wait for launch completion");
    launches.add_permits(1);
    drain.await.unwrap().unwrap();
    assert_eq!(handle.snapshot().active_jobs, 0);
    handle.send(WorkerCommand::Close).unwrap();
    task.await.unwrap().unwrap();
}

#[tokio::test(start_paused = true)]
async fn registration_failure_closes_the_transport_and_emits_stopped() {
    let RuntimeRig { mut runtime, handle, peer, closed, sent: _sent, .. } = runtime_rig();
    peer.send(Ok(offer("not-a-registration"))).unwrap();
    let mut notifications = handle.subscribe();
    assert!(matches!(runtime.run().await, Err(RuntimeError::RetriesExhausted { retries: 0, .. })));
    assert_eq!(closed.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(matches!(notifications.recv().await.unwrap(), WorkerNotification::Stopped(_)));
    assert!(handle.send(WorkerCommand::Close).is_err());
}

#[tokio::test(start_paused = true)]
async fn close_during_registration_does_not_wait_for_the_peer() {
    let RuntimeRig { mut runtime, handle, peer: _peer, mut sent, closed, .. } = runtime_rig();
    let task = tokio::spawn(async move { runtime.run().await });
    sent.recv().await.expect("register must be sent");
    handle.send(WorkerCommand::Close).unwrap();
    tokio::time::timeout(Duration::from_secs(1), task).await.unwrap().unwrap().unwrap();
    assert_eq!(closed.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn closing_cancels_an_unfinished_dial() {
    let (mut runtime, handle) = Worker::with_components(
        Arc::new(HangingTransport::new(crate::transport::TransportKind::WebSocket)),
        runtime_cfg(0),
        WorkerConfig::default(),
        Arc::new(LoadHost::default()),
        Arc::new(TestClock(tokio::time::Instant::now())),
        blocking_io(),
    );
    let h = handle.clone();
    let driver = async { runtime.run().await };
    let closer = async {
        tokio::time::sleep(Duration::from_millis(1)).await;
        h.send(WorkerCommand::Close).unwrap();
    };
    let (result, ()) =
        tokio::time::timeout(Duration::from_secs(1), async { tokio::join!(driver, closer) })
            .await
            .unwrap();
    result.unwrap();
    assert_eq!(handle.snapshot().finished.as_deref(), Some("closed"));
}

// Catches: a worker with endpoints retrying forever, silently, against a server that does not
// accept WebTransport (LiveKit Cloud production), instead of failing startup with what to fix.
#[tokio::test(start_paused = true)]
async fn an_unanswered_webtransport_dial_fails_startup_with_the_reason() {
    let (mut runtime, _handle) = Worker::with_components(
        Arc::new(HangingTransport::new(crate::transport::TransportKind::WebTransport)),
        runtime_cfg(16),
        WorkerConfig::default(),
        Arc::new(LoadHost::default()),
        Arc::new(TestClock(tokio::time::Instant::now())),
        blocking_io(),
    );
    let result = tokio::time::timeout(Duration::from_secs(11), runtime.run()).await.unwrap();
    let error = result.unwrap_err();
    assert!(matches!(error, RuntimeError::WebTransportUnreachable { .. }), "{error:?}");
    assert!(error.to_string().contains("WebTransport (QUIC over UDP)"), "{error}");
}

#[tokio::test(start_paused = true)]
async fn dropping_command_handles_does_not_starve_registration_deadlines() {
    let RuntimeRig { mut runtime, handle, peer: _peer, sent: _sent, closed, .. } = runtime_rig();
    drop(handle);
    let result = tokio::time::timeout(Duration::from_secs(11), runtime.run()).await.unwrap();
    assert!(matches!(result, Err(RuntimeError::RetriesExhausted { retries: 0, .. })));
    assert_eq!(closed.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn write_failure_closes_connection_and_exposes_cause() {
    let RuntimeRig { mut runtime, handle, peer: _peer, sent, closed, .. } = runtime_rig();
    drop(sent);
    let error = runtime.run().await.unwrap_err().to_string();
    assert!(error.contains("channel closed"), "{error}");
    assert_eq!(handle.snapshot().finished.as_deref(), Some(error.as_str()));
    assert_eq!(closed.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(handle.send(WorkerCommand::Close).is_err());
}

#[tokio::test(start_paused = true)]
async fn dropping_runtime_notifies_waiters_and_closes_command_queue() {
    let RuntimeRig { runtime, handle, .. } = runtime_rig();
    let mut notifications = handle.subscribe();
    drop(runtime);
    assert_eq!(
        notifications.recv().await.unwrap(),
        WorkerNotification::Stopped(WorkerStopped { reason: "worker runtime cancelled".into() })
    );
    assert!(handle.snapshot().finished.is_some());
    assert!(handle.send(WorkerCommand::Close).is_err());
}

#[tokio::test(start_paused = true)]
async fn a_stalled_write_after_registration_expires_and_closes_the_connection() {
    let RuntimeRig { mut runtime, handle, peer, sent: _sent, closed, stall_writes, .. } =
        runtime_rig();
    let mut notifications = handle.subscribe();
    peer.send(register_message("worker")).unwrap();
    let task = tokio::spawn(async move { runtime.run().await });
    assert!(matches!(notifications.recv().await.unwrap(), WorkerNotification::Registered(_)));
    stall_writes.store(true, std::sync::atomic::Ordering::SeqCst);
    // A periodic status write must expire even while load/status timers continue firing.
    let error = tokio::time::timeout(Duration::from_secs(15), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err()
        .to_string();
    assert!(error.contains("control write deadline exceeded"), "{error}");
    assert_eq!(closed.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(handle.snapshot().finished.as_deref(), Some(error.as_str()));
}

#[tokio::test(start_paused = true)]
async fn disconnect_cancels_pending_host_decisions_before_reconnect_finishes() {
    let RuntimeRig { mut runtime, handle, peer, mut started, sent: _sent, dropped, .. } =
        runtime_rig_with_retry(5);
    peer.send(register_message("worker")).unwrap();
    peer.send(Ok(offer("pending"))).unwrap();
    let task = tokio::spawn(async move { runtime.run().await });
    assert_eq!(started.recv().await, Some("decision"));
    peer.send(Err(TransportError::Failed("peer disconnected".into()))).unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        while dropped.load(std::sync::atomic::Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("disconnect must cancel the outstanding decision");
    assert!(!task.is_finished(), "worker should still be reconnecting");
    assert_eq!(dropped.load(std::sync::atomic::Ordering::SeqCst), 1);
    handle.send(WorkerCommand::Close).unwrap();
    task.await.unwrap().unwrap();
}

#[tokio::test(start_paused = true)]
async fn worker_command_saturation_control() {
    let (mut worker, _, handle) = conn(vec![Step::Serves(vec![register_message("W")])], 0);
    for index in 0..64 {
        handle
            .send(WorkerCommand::JobStatus {
                job_id: format!("job-{index}"),
                status: protocol::JobStatus::JsRunning,
            })
            .unwrap();
    }
    assert!(
        handle
            .send(WorkerCommand::JobStatus {
                job_id: "overflow".into(),
                status: protocol::JobStatus::JsRunning,
            })
            .is_err()
    );
    assert!(
        handle.send(WorkerCommand::Drain { timeout: None }).is_ok(),
        "drain needs reserved admission"
    );
    assert!(handle.send(WorkerCommand::Close).is_ok(), "close needs reserved admission");
    worker.run().await.unwrap();
    assert_eq!(handle.snapshot().finished.as_deref(), Some("closed"));
}

#[tokio::test(start_paused = true)]
async fn worker_sampler_off_loop() {
    let (mut worker, _, handle) = conn(vec![Step::Serves(vec![register_message("ready")])], 0);
    let blocking_io = worker.blocking_io.clone();
    let mut permits = Vec::new();
    for _ in 0..4 {
        permits.push(blocking_io.try_admit(()).unwrap());
    }
    let run = worker.run();
    tokio::pin!(run);
    tokio::time::timeout(Duration::from_secs(1), async {
        while handle.snapshot().worker_id != "ready" {
            assert!(futures_util::poll!(&mut run).is_pending());
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("CPU admission must not stall registration");
    handle.send(WorkerCommand::Drain { timeout: None }).unwrap();
    handle.send(WorkerCommand::Close).unwrap();
    tokio::time::timeout(Duration::from_millis(2), run)
        .await
        .expect("close must not wait for a sampler admission")
        .unwrap();
    assert_eq!(handle.snapshot().drain_outcome, Some(DrainOutcome::Finished));
    drop(permits);
}

#[tokio::test(start_paused = true)]
async fn accepting_tcp_without_registering_does_not_restore_the_retry_budget() {
    let (mut worker, wire, _) = conn(vec![Step::Serves(vec![]), Step::Serves(vec![])], 1);
    let error = worker.run().await.unwrap_err();
    assert!(matches!(&error, RuntimeError::RetriesExhausted { retries: 1, .. }));
    assert_eq!(wire.lock().attempts, 2, "only registration restores the retry budget");
    assert!(error.to_string().contains("registration timed out"));
}

// Catches: skipping the 5-sample average, so one CPU spike marks a healthy worker as full.
#[test]
fn default_load_calc_averages_five_samples() {
    let mut c = crate::DefaultLoadCalc::new();
    assert_eq!(c.get(), 0.0, "no CPU intervals have been averaged yet");
    for s in [1.0, 0.0, 0.0, 0.0, 0.0] {
        c.add_sample(s, UPDATE_LOAD_INTERVAL);
    }
    assert!((c.get() - 0.2).abs() < 1e-6, "{}", c.get());
    // Window slides: the spike falls out after five more samples.
    for _ in 0..5 {
        c.add_sample(0.0, UPDATE_LOAD_INTERVAL);
    }
    assert_eq!(c.get(), 0.0);
}

#[path = "job_dispatch_tests.rs"]
mod job_dispatch;

mod generated_sequences {
    use super::*;

    // These sequences drive the async worker through its real command and transport queues.
    // Reusing the gated host keeps the same boundary behavior as the individual race regressions.

    async fn wait_for_snapshot(handle: &WorkerClient, predicate: impl Fn(&Snapshot) -> bool) {
        tokio::time::timeout(Duration::from_secs(1), async {
            while !predicate(&handle.snapshot()) {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("worker did not settle the generated operation");
    }

    async fn next_host_operation(
        started: &mut tokio::sync::mpsc::UnboundedReceiver<&'static str>,
    ) -> &'static str {
        tokio::time::timeout(Duration::from_secs(1), started.recv())
            .await
            .expect("host operation was not started")
            .expect("worker dropped the host")
    }

    fn record_reply(
        message: protocol::WorkerMessage,
        replies: &mut std::collections::BTreeMap<String, Vec<bool>>,
    ) {
        if let Some(livekit_protocol::worker_message::Message::Availability(reply)) =
            message.message
        {
            replies.entry(reply.job_id).or_default().push(reply.available);
        }
    }

    async fn receive_reply(
        sent: &mut tokio::sync::mpsc::UnboundedReceiver<protocol::WorkerMessage>,
        replies: &mut std::collections::BTreeMap<String, Vec<bool>>,
        job_id: &str,
    ) {
        tokio::time::timeout(Duration::from_secs(1), async {
            while !replies.contains_key(job_id) {
                record_reply(sent.recv().await.expect("worker stopped"), replies);
            }
        })
        .await
        .expect("offered job was not answered");
    }

    #[tokio::test(start_paused = true)]
    async fn generated_offer_assignment_and_drain_sequences_settle_once() {
        // Base-three digits choose immediate completion, completion before launch returns, or an
        // expired assignment. Each sequence also varies duplicate offers and assignment messages.
        for seed in 0..81_u64 {
            let RuntimeRig {
                mut runtime,
                handle,
                peer,
                mut sent,
                mut started,
                decisions,
                launches,
                ..
            } = runtime_rig();
            let mut notifications = handle.subscribe();
            let task = tokio::spawn(async move { runtime.run().await });
            let first = sent.recv().await.expect("register was not sent");
            assert!(matches!(
                first.message,
                Some(livekit_protocol::worker_message::Message::Register(_))
            ));
            peer.send(register_message("sequence-worker")).unwrap();
            assert!(matches!(
                notifications.recv().await.unwrap(),
                WorkerNotification::Registered(_)
            ));
            let mut replies = std::collections::BTreeMap::new();
            let mut choices = seed;
            for index in 0..4 {
                let mode = choices % 3;
                choices /= 3;
                let job_id = format!("seed-{seed}-job-{index}");
                decisions.add_permits(1);
                for _ in 0..=seed % 3 {
                    peer.send(Ok(offer(&job_id))).unwrap();
                }
                assert_eq!(next_host_operation(&mut started).await, "decision", "seed {seed}");
                receive_reply(&mut sent, &mut replies, &job_id).await;
                assert_eq!(replies[&job_id], [true], "seed {seed}");
                // Let the writer's completion reach the worker before advancing its assignment clock.
                tokio::time::sleep(Duration::from_millis(1)).await;
                if mode == 2 {
                    tokio::time::advance(crate::ASSIGNMENT_TIMEOUT).await;
                    wait_for_snapshot(&handle, |snapshot| snapshot.reserved_slots == 0).await;
                    for _ in 0..=seed % 2 {
                        peer.send(Ok(assignment_message(&job_id))).unwrap();
                    }
                    // A failed disposition establishes that the late assignment was consumed.
                    tokio::time::timeout(Duration::from_secs(1), async {
                        loop {
                            let message = sent.recv().await.expect("worker stopped");
                            let failed = matches!(&message.message,
                                Some(livekit_protocol::worker_message::Message::UpdateJob(update))
                                    if update.job_id == job_id
                                        && update.status == protocol::JobStatus::JsFailed as i32);
                            record_reply(message, &mut replies);
                            if failed {
                                break;
                            }
                        }
                    })
                    .await
                    .expect("late assignment was not refused");
                    assert!(
                        started.try_recv().is_err(),
                        "expired assignment launched: seed {seed}"
                    );
                    continue;
                }
                for _ in 0..=seed % 3 {
                    peer.send(Ok(assignment_message(&job_id))).unwrap();
                }
                assert_eq!(next_host_operation(&mut started).await, "launch", "seed {seed}");
                if mode == 1 {
                    handle
                        .send(WorkerCommand::JobStatus {
                            job_id: job_id.clone(),
                            status: protocol::JobStatus::JsSuccess,
                        })
                        .unwrap();
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
                launches.add_permits(1);
                wait_for_snapshot(&handle, |snapshot| snapshot.reserved_slots == 0).await;
                if mode == 0 {
                    assert_eq!(handle.snapshot().active_jobs, 1, "seed {seed}");
                    handle
                        .send(WorkerCommand::JobStatus {
                            job_id,
                            status: protocol::JobStatus::JsSuccess,
                        })
                        .unwrap();
                }
                wait_for_snapshot(&handle, |snapshot| snapshot.active_jobs == 0).await;
                assert!(started.try_recv().is_err(), "duplicate assignment launched: seed {seed}");
            }

            // Keep one accepted request pending when draining; its assignment timeout must release
            // the drain, and offers received during drain must never invoke the host or be accepted.
            let pending_id = format!("seed-{seed}-pending");
            decisions.add_permits(1);
            peer.send(Ok(offer(&pending_id))).unwrap();
            assert_eq!(next_host_operation(&mut started).await, "decision");
            receive_reply(&mut sent, &mut replies, &pending_id).await;
            tokio::time::sleep(Duration::from_millis(1)).await;
            handle.send(WorkerCommand::Drain { timeout: Some(Duration::from_secs(20)) }).unwrap();
            wait_for_snapshot(&handle, |snapshot| snapshot.draining).await;
            assert!(
                handle.snapshot().drain_outcome.is_none(),
                "drained with a pending job: seed {seed}"
            );
            let refused_id = format!("seed-{seed}-during-drain");
            peer.send(Ok(offer(&refused_id))).unwrap();
            receive_reply(&mut sent, &mut replies, &refused_id).await;
            assert_eq!(replies[&refused_id], [false], "seed {seed}");
            tokio::time::advance(crate::ASSIGNMENT_TIMEOUT).await;
            handle.drain(Some(Duration::from_secs(20))).await.unwrap();
            assert_eq!(handle.snapshot().reserved_slots, 0, "seed {seed}");
            assert_eq!(handle.snapshot().active_jobs, 0, "seed {seed}");
            handle.send(WorkerCommand::Close).unwrap();
            task.await.unwrap().unwrap();
            while let Ok(message) = sent.try_recv() {
                record_reply(message, &mut replies);
            }
            assert!(
                replies.values().all(|answers| answers.len() == 1),
                "duplicate reply: seed {seed}"
            );
            assert!(started.try_recv().is_err(), "draining worker invoked host: seed {seed}");
            let mut drain_finished = 0;
            while let Ok(event) = notifications.try_recv() {
                match event {
                    WorkerNotification::Drained(DrainOutcome::Finished) => drain_finished += 1,
                    WorkerNotification::Drained(DrainOutcome::TimedOut) => {
                        panic!("drain timed out: seed {seed}")
                    }
                    _ => {}
                }
            }
            assert_eq!(drain_finished, 1, "seed {seed}");
        }
    }
}
