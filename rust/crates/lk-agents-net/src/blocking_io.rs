pub use lk_agents_media::pool::{Job, JobPanicked};
use lk_agents_media::pool::{PerProcess, Pool};

// Separate from the media pool, so saturated payload work never blocks a lookup or a sample.

/// A single-class pool: `()` is its class.
pub type BlockingIo = Pool<()>;

/// Lookups, small file writes and samples are short; four threads keep one stalled resolver
/// from serializing the rest without letting blocking work outgrow the async workers.
const THREADS: usize = 4;

/// The process's blocking I/O pool, started on first use.
pub fn blocking_io() -> BlockingIo {
    static POOL: PerProcess<BlockingIo> = PerProcess::new(|| {
        Pool::with_limits("lk-blocking-io", THREADS, &[THREADS]).expect("positive limits")
    });
    POOL.get()
}
