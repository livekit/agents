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

fn workers(total: usize) -> MediaWorkers {
    MediaWorkers::new(MediaWorkerLimits {
        total,
        decoders: 1,
        recordings: 1,
        images: 1,
        formatting: 1,
        eot_models: 1,
        vad_models: 1,
        worker_io: 1,
    })
    .unwrap()
}

#[test]
fn capacity_notification_observes_released_reservations() {
    let workers = workers(3);
    let permit = workers.try_admit(MediaWorkerClass::Recording).unwrap();
    assert!(workers.try_admit(MediaWorkerClass::Recording).is_err());
    let mut available = workers.wait_for_capacity(MediaWorkerClass::Recording);
    assert!(pending(&mut available));
    drop(permit);
    block_on(available);
    let _admitted = workers.try_admit(MediaWorkerClass::Recording).unwrap();
}

#[test]
fn native_payload_saturation_reserves_health_sampling_capacity() {
    let workers = workers(4);
    let _decoder = workers.try_admit(MediaWorkerClass::Decoder).unwrap();
    let _recording = workers.try_admit(MediaWorkerClass::Recording).unwrap();
    let _image = workers.try_admit(MediaWorkerClass::Image).unwrap();
    assert!(matches!(
        workers.try_admit(MediaWorkerClass::EotModel),
        Err(MediaWorkerError::Exhausted(MediaWorkerClass::EotModel))
    ));
    let _health = workers.try_admit(MediaWorkerClass::WorkerIo).unwrap();
}

#[test]
fn a_dropped_receipt_does_not_release_a_running_job() {
    let workers = workers(1);
    let (release, wait) = mpsc::channel::<()>();
    let (started, ready) = mpsc::channel();
    let receipt = workers
        .try_spawn(MediaWorkerClass::Decoder, move || {
            started.send(()).unwrap();
            wait.recv().unwrap();
        })
        .unwrap();
    ready.recv().unwrap();
    drop(receipt);
    assert!(matches!(
        workers.try_spawn(MediaWorkerClass::Recording, || ()),
        Err(MediaWorkerError::Exhausted(MediaWorkerClass::Recording))
    ));
    release.send(()).unwrap();
    // Capacity is released by settlement, not by dropping the caller.
    block_on(workers.wait_for_capacity(MediaWorkerClass::Recording));
    block_on(workers.try_spawn(MediaWorkerClass::Recording, || ()).unwrap()).unwrap();
}

#[test]
fn a_queued_job_never_waits_behind_a_long_lived_one() {
    let workers = workers(3);
    let (release, wait) = mpsc::channel::<()>();
    let recording = workers.try_spawn(MediaWorkerClass::Recording, move || wait.recv()).unwrap();
    let image = workers.try_spawn(MediaWorkerClass::Image, || "prepared").unwrap();
    assert_eq!(block_on(image), Ok("prepared"));
    release.send(()).unwrap();
    block_on(recording).unwrap().unwrap();
}

#[test]
fn a_panicking_job_reports_and_releases_its_capacity() {
    let workers = workers(1);
    let receipt = workers.try_spawn(MediaWorkerClass::Decoder, || panic!("decoder")).unwrap();
    assert_eq!(block_on(receipt), Err::<(), _>(WorkerPanicked));
    block_on(workers.try_spawn(MediaWorkerClass::Decoder, || ()).unwrap()).unwrap();
}

#[test]
fn models_and_decoders_share_the_payload_ceiling() {
    let workers = workers(3);
    let (release_decoder, decoder_wait) = mpsc::channel::<()>();
    let (release_model, model_wait) = mpsc::channel::<()>();
    let decoder =
        workers.try_spawn(MediaWorkerClass::Decoder, move || decoder_wait.recv()).unwrap();
    let model = workers.try_spawn(MediaWorkerClass::EotModel, move || model_wait.recv()).unwrap();
    assert!(matches!(
        workers.try_spawn(MediaWorkerClass::Image, || ()),
        Err(MediaWorkerError::Exhausted(MediaWorkerClass::Image))
    ));
    release_decoder.send(()).unwrap();
    block_on(decoder).unwrap().unwrap();
    block_on(workers.try_spawn(MediaWorkerClass::Image, || ()).unwrap()).unwrap();
    release_model.send(()).unwrap();
    block_on(model).unwrap().unwrap();
}
