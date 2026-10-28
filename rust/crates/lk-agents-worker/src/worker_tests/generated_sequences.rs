use super::*;

// These sequences drive the async worker through its real command and transport queues.
// Reusing the gated host keeps the same boundary behavior as the individual race regressions.

async fn wait_for_snapshot(handle: &WorkerHandle, predicate: impl Fn(&Snapshot) -> bool) {
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
    if let Some(livekit_protocol::worker_message::Message::Availability(reply)) = message.message {
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
        assert!(matches!(notifications.recv().await.unwrap(), WorkerNotification::Registered(_)));
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
                assert!(started.try_recv().is_err(), "expired assignment launched: seed {seed}");
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
        assert!(replies.values().all(|answers| answers.len() == 1), "duplicate reply: seed {seed}");
        assert!(started.try_recv().is_err(), "draining worker invoked host: seed {seed}");
        let mut drain_finished = 0;
        while let Ok(event) = notifications.try_recv() {
            match event {
                WorkerNotification::DrainFinished(_) => drain_finished += 1,
                WorkerNotification::DrainTimedOut(_) => panic!("drain timed out: seed {seed}"),
                _ => {}
            }
        }
        assert_eq!(drain_finished, 1, "seed {seed}");
    }
}
