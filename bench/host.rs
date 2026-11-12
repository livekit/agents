//! The cost of the host boundary: a Rust caller's call into Python through `HostInvoker`'s entry,
//! the context it runs in, host streams in both directions and cancellation, each beside the
//! pure-Python crossing it replaces. Timings are host-dependent observations, never gates.
//!
//! `make bench-host`, or:
//! `uv run --no-sync cargo test --release --manifest-path rust/Cargo.toml -p lk-agents-py
//!  --features ffmpeg --test host_bench -- --ignored --nocapture`
#![cfg(test)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use _core::py_calls;
use _core::py_calls::{Called, Entry, Invocation};
use _core::py_streams::{Convert, HostStream, ToHost};
use lk_agents_abi::BoxFuture;
use lk_agents_abi::telemetry::{Cx, record::SpanContext};
use lk_agents_abi::utils::clock::Clock;
use lk_agents_abi::utils::host_ref::HostRef;
use lk_agents_abi::{StreamInput, StreamOutput};
use lk_agents_runtime::net::clock::RealClock;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use serde_json::{Value, json};

#[path = "../rust/crates/lk-agents-py/src/py_test_support.rs"]
pub mod py_test_support;

/// `host()` from `bench/host.py`, beside this file.
fn fixture(py: Python<'_>) -> Bound<'_, PyAny> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../bench/host.py");
    let source = std::ffi::CString::new(include_str!("host.py")).unwrap();
    let module = PyModule::from_code(
        py,
        &source,
        &std::ffi::CString::new(path).unwrap(),
        c"_lk_agents_host_bench",
    )
    .unwrap();
    module.getattr("host").unwrap().call0().unwrap()
}

const WARMUP: usize = 500;
const CALLS: usize = 20_000;
const CONCURRENT: usize = 32;
const ITEMS: usize = 20_000;
/// A 20ms mono frame at 48kHz, 16-bit.
const FRAME_BYTES: usize = 1_920;
const CANCELS: usize = 500;

struct Host(Py<PyAny>);

impl Drop for Host {
    fn drop(&mut self) {
        Python::attach(|py| {
            let _ = self.0.bind(py).getattr("close").and_then(|close| close.call0());
        });
    }
}

impl Host {
    fn get(&self, name: &str) -> Py<PyAny> {
        Python::attach(|py| self.0.bind(py).getattr(name).unwrap().unbind())
    }

    fn context(&self, depth: usize) -> HostRef {
        Python::attach(|py| {
            let pair = self.0.bind(py).getattr("capture").unwrap().call1((depth,)).unwrap();
            _core::py_objects::retain(pair.unbind())
        })
    }
}

/// One host call through the production entry: `callable()` in a copy of `context`, under `cx`.
fn start(context: &HostRef, cx: &Cx, callable: &Py<PyAny>) -> BoxFuture<'static, PyResult<Called>> {
    let callable = Python::attach(|py| callable.clone_ref(py));
    lk_agents_runtime::host::call(&Entry, "bench", context, &[], cx, Box::new(invocation(callable)))
}

fn invocation(
    callable: Py<PyAny>,
) -> impl for<'py> FnOnce(Python<'py>) -> PyResult<Invocation<'py>> + Send + 'static {
    move |py| Ok((callable.into_bound(py), PyTuple::empty(py), None))
}

async fn call(context: &HostRef, cx: &Cx, callable: &Py<PyAny>) {
    start(context, cx, callable).await.unwrap();
}

/// Per-operation latencies in nanoseconds, as a report.
fn report(mut samples: Vec<u64>) -> Value {
    samples.sort_unstable();
    let at = |q: f64| samples[((samples.len() as f64 * q).ceil() as usize).saturating_sub(1)];
    let mean = samples.iter().sum::<u64>() as f64 / samples.len() as f64;
    json!({
        "count": samples.len(),
        "mean_us": mean / 1e3,
        "p50_us": at(0.5) as f64 / 1e3,
        "p99_us": at(0.99) as f64 / 1e3,
        "p999_us": at(0.999) as f64 / 1e3,
        "max_us": *samples.last().unwrap() as f64 / 1e3,
    })
}

async fn timed_calls(context: &HostRef, cx: &Cx, callable: &Py<PyAny>) -> Value {
    for _ in 0..WARMUP {
        call(context, cx, callable).await;
    }
    let mut samples = Vec::with_capacity(CALLS);
    for _ in 0..CALLS {
        let started = Instant::now();
        call(context, cx, callable).await;
        samples.push(started.elapsed().as_nanos() as u64);
    }
    report(samples)
}

/// Nanoseconds per call of the same work crossed from a foreign Python thread.
fn baseline(host: &Host, kind: &str) -> f64 {
    Python::attach(|py| {
        let f = host.0.bind(py).getattr("threadsafe_baseline").unwrap();
        f.call1((WARMUP, kind)).unwrap();
        f.call1((CALLS, kind)).unwrap().extract().unwrap()
    })
}

fn with_baseline(mut timed: Value, baseline_ns: f64) -> Value {
    let mean_us = timed["mean_us"].as_f64().unwrap();
    timed["python_threadsafe_mean_us"] = json!(baseline_ns / 1e3);
    timed["ratio_to_python"] = json!(mean_us * 1e3 / baseline_ns);
    timed
}

async fn calls(host: &Host) -> Value {
    let context = host.context(0);
    let deep = host.context(64);
    let span = SpanContext { trace_id: 0x5eed, span_id: 0x5eed, sampled: true };
    let in_span = Cx::NONE.continuing(Some(span));
    let (sync, noop, yielding) =
        (host.get("sync_noop"), host.get("async_noop"), host.get("async_yield"));
    json!({
        "sync": with_baseline(
            timed_calls(&context, &Cx::NONE, &sync).await, baseline(host, "sync")),
        "async": with_baseline(
            timed_calls(&context, &Cx::NONE, &noop).await, baseline(host, "async")),
        "async_one_await": with_baseline(
            timed_calls(&context, &Cx::NONE, &yielding).await, baseline(host, "yield")),
        "sync_in_span": timed_calls(&context, &in_span, &sync).await,
        "sync_64_contextvars": timed_calls(&deep, &Cx::NONE, &sync).await,
        "sync_64_contextvars_in_span": timed_calls(&deep, &in_span, &sync).await,
    })
}

/// `CONCURRENT` callers sharing one loop: throughput, and how late the loop's 1ms heartbeat ran.
async fn concurrent(host: &Host) -> Value {
    let context = Arc::new(host.context(0));
    let callable = Arc::new(host.get("async_yield"));
    let lag = host.get("lag");
    Python::attach(|py| {
        lag.bind(py).call_method0("start").unwrap();
    });
    let per_caller = CALLS / CONCURRENT;
    let started = Instant::now();
    let mut callers = tokio::task::JoinSet::new();
    for _ in 0..CONCURRENT {
        let (context, callable) = (context.clone(), callable.clone());
        callers.spawn(async move {
            let mut samples = Vec::with_capacity(per_caller);
            for _ in 0..per_caller {
                let at = Instant::now();
                call(&context, &Cx::NONE, &callable).await;
                samples.push(at.elapsed().as_nanos() as u64);
            }
            samples
        });
    }
    let mut samples = Vec::with_capacity(CALLS);
    while let Some(caller) = callers.join_next().await {
        samples.extend(caller.unwrap());
    }
    let elapsed = started.elapsed();
    let lag: Vec<u64> =
        Python::attach(|py| lag.bind(py).call_method0("stop").unwrap().extract().unwrap());
    json!({
        "callers": CONCURRENT,
        "calls_per_second": samples.len() as f64 / elapsed.as_secs_f64(),
        "latency": report(samples),
        "loop_lag": report(lag),
    })
}

fn open<T: Send + 'static>(host: &Host, size: usize, convert: Convert<T>) -> HostStream<T> {
    let context = host.context(0);
    Python::attach(|py| {
        let iterable = host.0.bind(py).getattr("stream").unwrap().call1((ITEMS, size)).unwrap();
        HostStream::open(py, &iterable, &context, convert, Arc::new(RealClock::new())).unwrap()
    })
}

async fn pull_all<T: Send + 'static>(mut stream: HostStream<T>) -> Value {
    let mut samples = Vec::with_capacity(ITEMS);
    let started = Instant::now();
    loop {
        let at = Instant::now();
        if stream.next().await.unwrap().is_none() {
            break;
        }
        samples.push(at.elapsed().as_nanos() as u64);
    }
    let elapsed = started.elapsed();
    let deadline = RealClock::new().now().saturating_add(Duration::from_secs(2));
    let at = Instant::now();
    Box::new(stream).close(deadline).await;
    let mut pulled = report(samples);
    pulled["items_per_second"] = json!(ITEMS as f64 / elapsed.as_secs_f64());
    pulled["close_us"] = json!(at.elapsed().as_nanos() as f64 / 1e3);
    pulled
}

/// Pushes through a duplex stream's input: what admission costs the caller, and how long until
/// the host took every item.
async fn push_all(host: &Host) -> Value {
    let context = host.context(0);
    let (sink, mut output) = Python::attach(|py| {
        let target = host.0.bind(py).getattr("Sink").unwrap().call0().unwrap();
        let to_host: ToHost<u64> = |py, n| pyo3::IntoPyObjectExt::into_bound_py_any(n, py);
        let convert: Convert<String> = |o| Ok(o.str()?.to_string());
        HostStream::duplex(py, &target, &context, to_host, convert, Arc::new(RealClock::new()))
            .unwrap()
            .split()
    });
    let mut refused = 0usize;
    let mut samples = Vec::with_capacity(ITEMS);
    let started = Instant::now();
    for n in 0..ITEMS as u64 {
        let at = Instant::now();
        // A full queue refuses rather than waits: count it, and give the loop a turn.
        while sink.push(n).is_err() {
            refused += 1;
            tokio::task::yield_now().await;
        }
        samples.push(at.elapsed().as_nanos() as u64);
    }
    sink.end_input();
    let received: usize = output.next().await.unwrap().unwrap().parse().unwrap();
    let elapsed = started.elapsed();
    assert_eq!(received, ITEMS);
    let mut pushed = report(samples);
    pushed["items_per_second"] = json!(ITEMS as f64 / elapsed.as_secs_f64());
    pushed["refused_pushes"] = json!(refused);
    pushed
}

/// From dropping an entered call to its host task having run its `finally`.
async fn cancellation(host: &Host) -> Value {
    let context = host.context(0);
    let blocked = host.get("blocked");
    let (entered, cancelled) = (host.get("entered"), host.get("cancelled"));
    let mut samples = Vec::with_capacity(CANCELS);
    for _ in 0..CANCELS {
        Python::attach(|py| {
            entered.bind(py).call_method0("clear").unwrap();
            cancelled.bind(py).call_method0("clear").unwrap();
        });
        let pending = tokio::spawn(start(&context, &Cx::NONE, &blocked));
        let wait = |event: &Py<PyAny>| {
            Python::attach(|py| {
                let set = event.bind(py).call_method1("wait", (2.0,)).unwrap();
                assert!(set.is_truthy().unwrap());
            })
        };
        wait(&entered);
        let at = Instant::now();
        pending.abort();
        wait(&cancelled);
        samples.push(at.elapsed().as_nanos() as u64);
    }
    report(samples)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "host-dependent timings; run with --ignored --nocapture"]
async fn host_boundary() {
    py_test_support::install_package(_core::register);
    let host = Host(Python::attach(|py| fixture(py).unbind()));
    let token: Convert<String> = |o| Ok(o.str()?.to_string());
    let frame: Convert<Vec<u8>> = |o| o.extract();
    let results = json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "python": Python::version_str(),
        "calls": calls(&host).await,
        "concurrent": concurrent(&host).await,
        "stream_pull_tokens": pull_all(open(&host, 0, token)).await,
        "stream_pull_frames": pull_all(open(&host, FRAME_BYTES, frame)).await,
        "sink_push": push_all(&host).await,
        "cancellation": cancellation(&host).await,
    });
    println!("{}", serde_json::to_string_pretty(&results).unwrap());
    println!("HOST_BENCH {results}");
}
