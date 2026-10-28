use super::*;

#[tokio::test(start_paused = true)]
async fn generated_exit_retry_cancel_and_drain_sequences_release_every_process() {
    for seed in 0..48_u64 {
        let phase = std::cell::Cell::new("warmup");
        tokio::time::timeout(Duration::from_secs(30), exercise(seed, &phase))
            .await
            .unwrap_or_else(|_| panic!("sequence stalled: seed {seed}, phase {}", phase.get()));
    }
}

async fn exercise(seed: u64, phase: &std::cell::Cell<&'static str>) {
    let reaped = watch::channel(false).0;
    let host = Arc::new(Host {
        fail_launch: AtomicUsize::new((seed % 3) as usize),
        reap_gate: Some(reaped.clone()),
        ..Host::default()
    });
    let pool = pool(&host, 1 + (seed % 3) as usize);
    pool.start().await.unwrap();
    phase.set("observe warm exit");

    // Exit is visible before the receipt. The exited process remains owned during these
    // launches, but it must already have lost its eligibility to receive a job.
    host.close(&0).await.unwrap();
    tokio::time::sleep(REFILL_INTERVAL).await;
    assert!(pool.processes().contains(&0), "lost the reaping owner: seed {seed}");
    let mut requests = Vec::new();
    for job in 0..3 {
        let pool = pool.clone();
        requests.push(tokio::spawn(async move {
            pool.launch_job(&format!("seed-{seed}-job-{job}"), ()).await
        }));
    }
    tokio::time::sleep(Duration::from_millis(1)).await;
    phase.set("duplicate admission");
    assert!(
        matches!(
            pool.launch_job(&format!("seed-{seed}-job-0"), ()).await,
            Err(LaunchError::Admission(PoolAdmissionError::DuplicateJob))
        ),
        "duplicate admitted: seed {seed}"
    );

    // Sometimes cancel an admitted request while an earlier failed launch is still waiting
    // for its process's receipt. The other requests must continue on fresh processes.
    let cancel = seed % 2 == 1 && !requests[1].is_finished();
    if cancel {
        requests[1].abort();
    }
    for _ in 0..seed % 4 {
        tokio::task::yield_now().await;
    }
    phase.set("settle launches");
    reaped.send_replace(true);
    for (job, request) in requests.into_iter().enumerate() {
        let result = tokio::time::timeout(Duration::from_secs(4), request)
            .await
            .unwrap_or_else(|_| panic!("admitted request did not settle: seed {seed} job {job}"));
        if cancel && job == 1 {
            assert!(result.unwrap_err().is_cancelled());
        } else {
            result.unwrap().unwrap();
        }
    }
    assert_eq!(pool.jobs_waiting(), 0, "seed {seed}");

    phase.set("drain");
    let drain = tokio::spawn({
        let pool = pool.clone();
        async move { pool.join_jobs().await }
    });
    tokio::time::sleep(Duration::from_millis(1)).await;
    assert!(!drain.is_finished(), "drain finished with running jobs: seed {seed}");
    assert!(matches!(
        pool.launch_job("during-drain", ()).await,
        Err(LaunchError::Admission(PoolAdmissionError::Draining))
    ));
    let mut processes = pool.processes();
    if seed % 2 == 0 {
        processes.reverse();
    }
    for process in processes {
        host.close(&process).await.unwrap();
        if seed % 3 == 0 {
            tokio::task::yield_now().await;
        }
    }
    tokio::time::timeout(Duration::from_secs(4), drain).await.unwrap().unwrap();

    phase.set("close");
    let first_close = tokio::spawn({
        let pool = pool.clone();
        async move { pool.aclose().await }
    });
    if seed % 2 == 1 {
        tokio::task::yield_now().await;
        first_close.abort();
    }
    tokio::time::timeout(Duration::from_secs(4), pool.aclose()).await.unwrap();
    let _ = first_close.await;
    assert!(pool.processes().is_empty(), "resources retained after cleanup: seed {seed}");

    phase.set("verify events");
    let mut retired = std::collections::BTreeSet::new();
    let mut launched = std::collections::BTreeSet::new();
    let events = host.events.lock().unwrap();
    for &(event, process) in events.iter() {
        match event {
            PoolEvent::Closed => {
                assert!(retired.insert(process), "process released twice: seed {seed}");
            }
            PoolEvent::Ready | PoolEvent::JobLaunched => {
                assert!(!retired.contains(&process), "released process revived: seed {seed}");
                if event == PoolEvent::JobLaunched {
                    assert_ne!(process, 0, "exited warm process launched: seed {seed}");
                    assert!(launched.insert(process), "process served two jobs: seed {seed}");
                }
            }
            _ => {}
        }
    }
    assert_eq!(launched.len(), 3 - usize::from(cancel), "seed {seed}");
    assert_eq!(retired.len(), host.next.load(Ordering::SeqCst), "process leaked: seed {seed}");
}
