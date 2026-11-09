#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "media's worker pool is the one threaded module of the sans-I/O cores"
)]

use std::sync::mpsc;
use std::task::Wake;
use std::thread::Thread;

use super::*;

struct Unpark(Thread);

impl Wake for Unpark {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut cx) {
            return output;
        }
        std::thread::park();
    }
}

fn pending<F: Future + Unpin>(future: &mut F) -> bool {
    Pin::new(future).poll(&mut Context::from_waker(Waker::noop())).is_pending()
}

fn media_pool(total: usize) -> MediaPool {
    MediaPool::new(MediaPoolLimits {
        total,
        decoders: 1,
        recordings: 1,
        images: 1,
        formatting: 1,
        eot_models: 1,
        vad_models: 1,
    })
    .unwrap()
}

#[test]
fn capacity_notification_observes_released_reservations() {
    let media_pool = media_pool(3);
    let permit = media_pool.try_admit(JobClass::Recording).unwrap();
    assert!(media_pool.try_admit(JobClass::Recording).is_err());
    let mut available = media_pool.wait_for_capacity(JobClass::Recording);
    assert!(pending(&mut available));
    drop(permit);
    block_on(available);
    let _admitted = media_pool.try_admit(JobClass::Recording).unwrap();
}

#[test]
fn a_dropped_receipt_does_not_release_a_running_job() {
    let media_pool = media_pool(1);
    let (release, wait) = mpsc::channel::<()>();
    let (started, ready) = mpsc::channel();
    let receipt = media_pool
        .try_spawn(JobClass::Decoder, move || {
            started.send(()).unwrap();
            wait.recv().unwrap();
        })
        .unwrap();
    ready.recv().unwrap();
    drop(receipt);
    assert!(matches!(
        media_pool.try_spawn(JobClass::Recording, || ()),
        Err(MediaPoolError::Exhausted(JobClass::Recording))
    ));
    release.send(()).unwrap();
    // Capacity is released by settlement, not by dropping the caller.
    block_on(media_pool.wait_for_capacity(JobClass::Recording));
    block_on(media_pool.try_spawn(JobClass::Recording, || ()).unwrap()).unwrap();
}

#[test]
fn a_queued_job_never_waits_behind_a_long_lived_one() {
    let media_pool = media_pool(3);
    let (release, wait) = mpsc::channel::<()>();
    let recording = media_pool.try_spawn(JobClass::Recording, move || wait.recv()).unwrap();
    let image = media_pool.try_spawn(JobClass::Image, || "prepared").unwrap();
    assert_eq!(block_on(image), Ok("prepared"));
    release.send(()).unwrap();
    block_on(recording).unwrap().unwrap();
}

#[test]
fn a_panicking_job_reports_and_releases_its_capacity() {
    let media_pool = media_pool(1);
    let receipt = media_pool.try_spawn(JobClass::Decoder, || panic!("decoder")).unwrap();
    assert_eq!(block_on(receipt), Err::<(), _>(JobLost));
    block_on(media_pool.try_spawn(JobClass::Decoder, || ()).unwrap()).unwrap();
}

#[test]
fn models_and_decoders_share_the_total_ceiling() {
    let media_pool = media_pool(2);
    let (release_decoder, decoder_wait) = mpsc::channel::<()>();
    let (release_model, model_wait) = mpsc::channel::<()>();
    let decoder = media_pool.try_spawn(JobClass::Decoder, move || decoder_wait.recv()).unwrap();
    let model = media_pool.try_spawn(JobClass::EotModel, move || model_wait.recv()).unwrap();
    assert!(matches!(
        media_pool.try_spawn(JobClass::Image, || ()),
        Err(MediaPoolError::Exhausted(JobClass::Image))
    ));
    release_decoder.send(()).unwrap();
    block_on(decoder).unwrap().unwrap();
    block_on(media_pool.try_spawn(JobClass::Image, || ()).unwrap()).unwrap();
    release_model.send(()).unwrap();
    block_on(model).unwrap().unwrap();
}

/// A pool no thread serves: made by `pid`, counting `used` of its one slot and an idle thread
/// that is gone, as a forked child inherits its parent's.
fn threadless(pid: u32, used: usize) -> MediaPool {
    let state = State {
        used: vec![used; 6].into(),
        total: used,
        tasks: VecDeque::new(),
        idle: 1,
        capacity_waiters: Vec::new(),
    };
    let shared = Shared {
        pid,
        thread_name: "lk-media-test",
        total: 1,
        limits: vec![1; 6].into(),
        state: Mutex::new(state),
        work: Condvar::new(),
    };
    Pool { shared: Arc::new(shared), class: PhantomData }
}

fn settled<T: Send + 'static>(job: impl Future<Output = T> + Send + 'static) -> T {
    let (done, settled) = mpsc::channel();
    std::thread::spawn(move || done.send(block_on(job)));
    settled.recv_timeout(Duration::from_secs(5)).expect("the waiter never resolved")
}

#[test]
fn a_job_dropped_unrun_settles_lost() {
    let pool = threadless(std::process::id(), 0);
    let job = pool.try_spawn(JobClass::Decoder, || 1).unwrap();
    let unrun = std::mem::take(&mut pool.shared.lock().tasks);
    drop(unrun);
    assert_eq!(settled(job), Err(JobLost));
}

#[test]
fn a_forked_child_never_waits_on_its_parents_pool() {
    let parent = std::process::id().wrapping_add(1);
    let queued = threadless(parent, 0).try_spawn(JobClass::Decoder, || 1).unwrap();
    assert_eq!(settled(queued), Err(JobLost));
    let full = threadless(parent, 1);
    assert_eq!(settled(async move { full.run(JobClass::Decoder, || 1).await }), Err(JobLost));
}
